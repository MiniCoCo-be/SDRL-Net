import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import jaccard_score
from torch import optim
from tqdm import tqdm
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms

from optparse import OptionParser

from torch.utils.tensorboard import SummaryWriter
import time
import math
import os
import sys
import pdb
import warnings
import logging

from Base.datasets.dataset_monuseg import ValGenerator, ImageToImage2D

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR + "-------------------------------")
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

from model.network_scunet import SCUNet

from model.CoMsnet import CoMsNet

from UT_base.utnet import UTNet
from utils.utils import *
from utils import metrics

from torch.utils import data
from losses import DiceLoss2, test_single_volume
from dataset_synapse import Synapse_dataset, RandomGenerator

warnings.filterwarnings("ignore", category=UserWarning)
DEBUG = False


def inference(args, model, test_save_path=None):
    logging.basicConfig(filename=args.cp_path + "/conv_att149test_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    tf_test = ValGenerator(output_size=[args.img_size, args.img_size])
    test_dataset = ImageToImage2D(args.test_dataset, tf_test, image_size=args.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    logging.info("{} test iterations per epoch".format(len(test_loader)))
    model.eval()

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(test_loader)):
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255

            input_img = torch.from_numpy(arr)
            model.eval()

            output = model(input_img.cuda())
            pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
            predict_save = pred_class[0].cpu().data.numpy()
            predict_save = np.reshape(predict_save, (args.img_size, args.img_size))
            tmp_lbl = (lab).astype(np.float32)
            tmp_3dunet = (predict_save).astype(np.float32)
            dice_pred_t = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
            # dice_show = "%.3f" % (dice_pred)
            iou_pred_t = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))

            dice_pred += dice_pred_t
            iou_pred += iou_pred_t

        dice_mean = dice_pred / len(test_dataset)
        iou_mean = iou_pred / len(test_dataset)
        print("dice_pred", dice_pred / len(test_dataset))
        print("iou_pred", iou_pred / len(test_dataset))


    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return dice_mean, iou_mean


if __name__ == '__main__':
    parser = OptionParser()


    def get_comma_separated_int_args(option, opt, value, parser):
        value_list = value.split(',')
        value_list = [int(i) for i in value_list]
        setattr(parser.values, option.dest, value_list)


    parser.add_option('--is_savenii', action="store_true", help='whether to save results during inference')
    parser.add_option('--volume_path', type=str, default='./Dataset/Synapse/test_vol_h5',
                      help='root dir for validation volume data')  # for acdc volume_path=root_dir)
    # 默认迭代150次
    parser.add_option('-e', '--epochs', dest='epochs', default=50, type='int', help='number of epochs')
    # 默认batch_size = 24
    parser.add_option('-b', '--batch_size', dest='batch_size', default=6, type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01, type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False, help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path', default='./checkpoint/',
                      help='checkpoint path')
    parser.add_option('--list_dir', type=str, default='./Base/lists/lists_Synapse', help='list dir')
    parser.add_option('--root_path', type=str, default='./Dataset/Synapse/train_npz', help='root dir for data')

    parser.add_option('-o', '--log-path', type='str', dest='log_path', default='./log/test/', help='log path')
    parser.add_option('-m', type='str', dest='model', default='CoMsNet', help='use which model')
    parser.add_option('--num_class', type='int', dest='num_class', default=9, help='number of segmentation classes')
    parser.add_option('--base_chan', type='int', dest='base_chan', default=32,
                      help='number of channels of first expansion in UNet')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name', default='test',
                      help='unique experiment name')
    parser.add_option('--rlt', type='float', dest='rlt', default=1, help='relation between CE/FL and dice')
    parser.add_option('--weight', type='float', dest='weight',
                      default=[0.5, 1, 1, 1], help='weight each class in loss function')
    parser.add_option('--weight_decay', type='float', dest='weight_decay',
                      default=0.0001)
    parser.add_option('--crop_size', type='int', dest='crop_size', default=224)
    parser.add_option('--domain', type='str', dest='domain', default='A')
    parser.add_option('--aux_weight', type='float', dest='aux_weight', default=[1, 0.4, 0.2, 0.1])
    parser.add_option('--reduce_size', dest='reduce_size', default=8, type='int')
    parser.add_option('--block_list', dest='block_list', default='1234', type='str')
    parser.add_option('--num_blocks', dest='num_blocks', default=[1, 1, 1, 1], type='string', action='callback',
                      callback=get_comma_separated_int_args)
    parser.add_option('--aux_loss', dest='aux_loss', action='store_true', help='using aux loss for deep supervision')
    parser.add_option('--seed', type=int, default=1234, help='random seed')

    parser.add_option('--gpu', type='str', dest='gpu', default='0')
    options, args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

    print('Using model:', options.model)

    if options.model == 'UTNet':
        net = UTNet(1, options.base_chan, options.num_class, reduce_size=options.reduce_size,
                    block_list=options.block_list, num_blocks=options.num_blocks, num_heads=[4, 4, 4, 4],
                    projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=options.aux_loss,
                    maxpool=True)
    if options.model == 'CoMsNet':
        net = CoMsNet(3, options.base_chan, options.num_class, num_heads=[3, 6, 12, 24], num_block=0,
                      num_croblock=[0, 0, 0, 0],
                      attn_drop=0.1, maxpool=True)
    if options.model == 'SCUNet':  # 默认224
        # print()
        net = SCUNet(in_nc=3, numclass=options.num_class)

    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))

    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # print(os.getcwd()+"====================")
    # print("sys.path----------------:",sys.path)
    os.chdir("/home/lxt/exp_pre/pro_run/exp_run/")
    # print("已==================修改当前目录：",os.getcwd())

    # print(net)
    # print(param_num)

    net.cuda()

    options.z_spacing = 1
    if options.is_savenii:
        options.test_save_dir = '../predictions'
        test_save_path = os.path.join(options.test_save_dir, options.exp)
        os.makedirs(test_save_path, exist_ok=True)
    else:
        test_save_path = None
    # 加载训练好的模型
    # pre_dir = os.path.join(options.cp_path, options.unique_name)
    # pre_list = os.listdir(pre_dir)
    # pre_path = pre_list[len(pre_list)-1]
    # path = os.path.join(pre_dir, pre_path)
    # assert os.path.exists(path), "{} path does not exist.".format(path)
    # net.load_state_dict(torch.load(os.path.join(pre_dir,pre_path))) /home/xys/lxt/exp_run/checkpoint/CoMs/146_sc_0__best.pth
    net.load_state_dict(torch.load("/home/lxt/exp_pre/pro_run/exp_run/checkpoint/Comsnet_conv/144_sc_0__best.pth"))
    # print(os.path.join(pre_dir,pre_path)+"---------------------------")
    # 之前/checkpoint/test/44_sc_0__best.pth-

    dice_mean, HD_mean = inference(options, net, test_save_path)
    torch.save(net.state_dict(), '%s%s/bestmodel_150_coms_conv_outtest.pth' % (options.cp_path, options.unique_name))


    print('done')

    sys.exit(0)
