import torch
import torch.nn as nn
import torch.nn.functional as F
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


#
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(BASE_DIR+"-------------------------------")
# # __file__获取执行文件相对路径，整行为取上一级的上一级目录
# sys.path.append(BASE_DIR)

# from model.Coms_conv import CoMs_conv
# from model.network_scunet import SCUNet
# from model.CoMsnet1 import CoMsNet1
#
#
#
# from model.CoMsnet import CoMsNet

# from UT_base.utnet import UTNet
from UT_tr.utils.utils import *

from Base.datasets.dataset_synapse import Synapse_dataset
from UT_base.utnet import UTNet
from model.Comsnet import CoMsNet
from model.CoMs_spach import CoMs_msspa_ch
from utils import metrics
from model.newest_copsp import Conewpsp
from torch.utils import data
from UT_tr.losses import DiceLoss2, test_single_volume
# from dataset_synapse import Synapse_dataset,RandomGenerator


warnings.filterwarnings("ignore", category=UserWarning)
DEBUG = False



def inference(args, model, test_save_path=None):
    logging.basicConfig(filename=args.cp_path + "/conv_att108test_log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    db_test = Synapse_dataset(base_dir=args.volume_path, split="test_vol", list_dir=args.list_dir)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    model.eval()
    metric_list = 0.0

    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch["image"].size()[2:]
        image, label, case_name = sampled_batch["image"], sampled_batch["label"], sampled_batch['case_name'][0]
        # 测试
        metric_i = test_single_volume(image, label, model, classes=args.num_class, patch_size=[args.crop_size, args.crop_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        metric_list += np.array(metric_i)
        # print('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
        logging.info('idx %d case %s mean_dice %f mean_hd95 %f' % (i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1]))
    metric_list = metric_list / len(db_test)
    for i in range(1, args.num_class):
        # print('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
        logging.info('Mean class %d mean_dice %f mean_hd95 %f' % (i, metric_list[i-1][0], metric_list[i-1][1]))
    performance = np.mean(metric_list, axis=0)[0]
    mean_hd95 = np.mean(metric_list, axis=0)[1]
    # print('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f' % (performance, mean_hd95))
    return performance, mean_hd95


if __name__ == '__main__':
    parser = OptionParser()
    def get_comma_separated_int_args(option, opt, value, parser):
        value_list = value.split(',')
        value_list = [int(i) for i in value_list]
        setattr(parser.values, option.dest, value_list)

    parser.add_option('--is_savenii', action="store_true", help='whether to save results during inference')
    parser.add_option('--volume_path', type=str, default='../Dataset/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir)
    # 默认迭代150次
    parser.add_option('-e', '--epochs', dest='epochs', default=50, type='int', help='number of epochs')
    # 默认batch_size = 24
    parser.add_option('-b', '--batch_size', dest='batch_size', default=6, type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01, type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False, help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path', default='D:\\Program Files\\vscode_code\\myUnet1\\new_copsplog/checkpoint/', help='checkpoint path')
    parser.add_option('--list_dir', type=str, default='../Base/lists/lists_Synapse', help='list dir')
   
    parser.add_option('-o', '--log-path', type='str', dest='log_path', default='D:\\Program Files\\vscode_code\\myUnet1\\new_copsplog\\log/', help='log path')
    parser.add_option('-m', type='str', dest='model', default='Conewpsp', help='use which model')
    parser.add_option('--num_class', type='int', dest='num_class', default=9, help='number of segmentation classes')
    parser.add_option('--base_chan', type='int', dest='base_chan', default=32, help='number of channels of first expansion in UNet')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name', default='testpsp', help='unique experiment name')
    parser.add_option('--rlt', type='float', dest='rlt', default=1, help='relation between CE/FL and dice')
    parser.add_option('--weight', type='float', dest='weight',
                      default=[0.5,1,1,1] , help='weight each class in loss function')
    parser.add_option('--weight_decay', type='float', dest='weight_decay',
                      default=0.0001)
    parser.add_option('--crop_size', type='int', dest='crop_size', default=224)
    parser.add_option('--domain', type='str', dest='domain', default='A')
    parser.add_option('--aux_weight', type='float', dest='aux_weight', default=[1, 0.4, 0.2, 0.1])
    parser.add_option('--reduce_size', dest='reduce_size', default=8, type='int')
    parser.add_option('--block_list', dest='block_list', default='1234', type='str')
    parser.add_option('--num_blocks', dest='num_blocks', default=[1,1,1,1], type='string', action='callback', callback=get_comma_separated_int_args)
    parser.add_option('--aux_loss', dest='aux_loss', action='store_true', help='using aux loss for deep supervision')
    parser.add_option('--seed', type=int, default=1234, help='random seed')
    
    parser.add_option('--gpu', type='str', dest='gpu', default='0')
    options, args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

    print('Using model:', options.model)

    if options.model == 'UTNet':  # CoMs_msspa_ch
        net = UTNet(1, options.base_chan, options.num_class, reduce_size=options.reduce_size, block_list=options.block_list, num_blocks=options.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=options.aux_loss, maxpool=True)
    if options.model == 'Conewpsp':
        net = Conewpsp(3, options.base_chan, options.num_class,num_heads=[3,6,12,24], num_block=0, num_croblock=[0,0,0,0],
               attn_drop=0.1, maxpool=True)
    if options.model == 'CoMsNet':
        net = CoMsNet(3, options.base_chan, options.num_class,num_heads=[3,6,12,24], num_block=0, num_croblock=[0,0,0,0],
               attn_drop=0.1, maxpool=True)
    if options.model == 'CoMs_msspa_ch':
        net = CoMs_msspa_ch(3, options.base_chan, options.num_class,num_heads=[3,6,12,24], num_block=0, num_croblock=[0,0,0,0],
               attn_drop=0.1, maxpool=True)
    # if options.model == 'CoMs_conv':
    #     net = CoMs_conv(3, options.base_chan, options.num_class,num_heads=[3,6,12,24], num_block=0, num_croblock=[0,0,0,0],
    #            attn_drop=0.1, maxpool=True)
        # net = CoMsNet(3, 32, 9,num_heads=[3,6,12,24], num_block=0, num_croblock=[0,0,0,0],
        #        attn_drop=0.1, maxpool=True)       
    # if options.model == 'SCUNet':  # 默认224
        # print()
        # net = SCUNet(in_nc=3, numclass=options.num_class)


    # if options.load:
    #     net.load_state_dict(torch.load(options.load))
    #     print('Model loaded from {}'.format(options.load))
    
    # param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    

    
    # print(os.getcwd()+"====================")
    # print("sys.path----------------:",sys.path)
    # os.chdir("/home/lxt/exp_pre/pro_run/exp_run/")
    # print("已==================修改当前目录：",os.getcwd())
    # print(net)

    # print(net)
    # print(param_num)
    
    # net.cuda()
    options.exp = "Synapse_testvis_"+ str(options.crop_size)
    options.z_spacing = 1
    if options.is_savenii:  
        options.test_save_dir = './prediction'
        test_save_path = os.path.join(options.test_save_dir, options.exp)
        os.makedirs(test_save_path, exist_ok=True)
        print("---------保存最后预测-----options.is_savenii-----path----",test_save_path)
    else:
        print("不保存模型")
        test_save_path = None
    # 加载训练好的模型
    # pre_dir = os.path.join(options.cp_path, options.unique_name)
    # pre_list = os.listdir(pre_dir)
    # pre_path = pre_list[len(pre_list)-1]
    # path = os.path.join(pre_dir, pre_path)
    # assert os.path.exists(path), "{} path does not exist.".format(path)
    # net.load_state_dict(torch.load(os.path.join(pre_dir,pre_path))) /home/xys/lxt/exp_run/checkpoint/CoMs/146_sc_0__best.pth
    # D:\\Program Files\\vscode_code\\myUnet1\\model\\deTup_pre\\147_sc_0__best.pth
    #原 ./spach_195_sc_0__best.pth
    net.load_state_dict(torch.load("D:\\Program Files\\vscode_code\\myUnet1\\model\\deTup_pre\\147_sc_0__best.pth",map_location=torch.device('cpu')))
    # /home/lxt/exp_pre/pro_run/exp_run/checkpoint/Comsnet_conv/134_sc_0__best.pth
    # print(os.path.join(pre_dir,pre_path)+"---------------------------")
    # 之前/checkpoint/test/44_sc_0__best.pth-
    
    # pre_weight1 = torch.load("/home/lxt/exp_pre/pro_run/exp_run/checkpoint/Comsnet_conv/134_sc_0__best.pth")
    # pre_dict = {k:v for k,v in pre_weight1.items() if "outc" not in k }
    # for k,v in pre_weight1.items():
    #      print("---------conv134-model-----k-------",net.state_dict()[k].numel() if k in net.state_dict() else k)
    #      print("-----para-v----",k,v.numel())

    dice_mean,  HD_mean = inference(options, net, test_save_path)
    
    # torch.save(net.state_dict(), '%s%s/bestmodel_108coms_att_conv_outtest.pth' % (options.cp_path, options.unique_name))

    # best_dice = 0.5
    # if dice_mean() <= best_dice:
    #     best_dice = dice_mean()
    #     torch.save(net.state_dict(), '%s%s/bestmodel.pth' % (options.cp_path, options.unique_name))

    # print('save done')
    # print('dice: %.5f/best dice: %.5f' % (dice_mean(), best_dice))

    print(' finishing  done')

    sys.exit(0)
