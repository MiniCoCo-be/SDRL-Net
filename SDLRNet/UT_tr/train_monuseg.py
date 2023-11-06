import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.cuda.amp import autocast as autocast
from optparse import OptionParser

from torch.utils.tensorboard import SummaryWriter
import time
import math
import os
import sys
import pdb
import warnings
import logging

from Base.datasets.dataset_monuseg import RandomGenerator, ValGenerator, ImageToImage2D
from UT_tr.losses import DiceLoss2
from UT_tr.monu_utils import WeightedDiceBCE
from model.Comsnet import CoMsNet

from sklearn.model_selection import cross_val_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR + "-------------------------------")
# __file__获取执行文件相对路径，整行为取上一级的上一级目录
sys.path.append(BASE_DIR)

warnings.filterwarnings("ignore", category=UserWarning)
DEBUG = False


def train_net(net, options,device):
    logging.basicConfig(filename=options.cp_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(options))

    def worker_init_fn(worker_id):
        torch.random.seed(options.seed + worker_id)

    train_tf = transforms.Compose([RandomGenerator(output_size=[options.img_size, options.img_size])])
    val_tf = ValGenerator(output_size=[options.img_size, options.img_size])
    train_dataset = ImageToImage2D(options.train_path, train_tf, image_size=options.img_size)
    val_dataset = ImageToImage2D(options.val_path, val_tf, image_size=options.img_size)
    train_loader = DataLoader(train_dataset,
                              batch_size=options.batch_size,
                              shuffle=True,
                              worker_init_fn=worker_init_fn,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset,
                            batch_size=options.batch_size,
                            shuffle=True,
                            worker_init_fn=worker_init_fn,
                      pin_memory=True)


    print("The length of train set is: {}".format(len(train_dataset)))
    print("The length of val set is: {}".format(len(val_dataset)))



    writer = SummaryWriter(options.log_path + options.unique_name)

    optimizer = optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=options.weight_decay)

    criterion = WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)

    max_iterations = options.epochs * len(train_loader)
    iter_num = 0
    base_lr = options.lr
    best_dice = 0
    best_epoch = 0

    for epoch in range(options.epochs):
        print('Starting epoch {}/{}'.format(epoch + 1, options.epochs))

        # exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=options.lr, epoch=epoch, warmup_epoch=5, max_epoch=options.epochs)
        loss_sum = 0
        dice_sum = 0.0
        for i_batch, (sampled_batch, names) in enumerate(train_loader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.to(device), label_batch.to(device)
            end = time.time()
            net.train()
            preds = net(image_batch)
            loss = criterion(preds, label_batch.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num += 1


        epoch_time = time.time() - end
        net.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(val_loader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.to(device), label_batch.to(device)
                preds = net(image_batch)

                val_dice = criterion._show_dice(preds, label_batch.float())
                dice_sum += val_dice
        val_dic = dice_sum / len(val_dataset)

        if os.path.isdir('%s%s/' % (options.cp_path, options.unique_name)):
            pass
        else:
            os.mkdir('%s%s/' % (options.cp_path, options.unique_name))

        if val_dic > best_dice:
           best_dice = val_dic
           best_epoch = epoch + 1
           torch.save(net.state_dict(),
                      '%s%s/%d_sc_%d__best.pth' % (options.cp_path, options.unique_name, best_epoch, best_dice))
           save_mode_path = os.path.join(options.cp_path, options.unique_name,
                                         str(epoch + 1) + "_sc_" + str(best_dice) + "_best.pth")
           logging.info("save model to {}, mean dice increased from: {:.4f} to {:.4f}".format(save_mode_path,best_dice,val_dice))


        print('[epoch %d] epoch loss: %.5f,epoch time: %.5f' % (epoch + 1, loss_sum / (i_batch + 1), epoch_time))
        writer.add_scalar('Train/Loss', loss_sum / (i_batch + 1), epoch + 1)
        writer.add_scalar('LR', lr_, epoch + 1)
        logging.info("epoch:{} Train_Loss:{} run_time:{}".format(epoch + 1, loss_sum / (i_batch + 1), epoch_time))
        logging.info("epoch:{} Train_LR:{}".format(epoch + 1, lr_))


    print("---------------training finishing---------------------")


if __name__ == '__main__':
    parser = OptionParser()

    # 默认迭代150次
    parser.add_option('-e', '--epochs', dest='epochs', default=150, type='int', help='number of epochs')
    # 默认batch_size = 24
    parser.add_option('-b', '--batch_size', dest='batch_size', default=6, type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01, type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False, help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path', default='./checkpoint/',
                      help='checkpoint path')
    parser.add_option('--list_dir', type=str, default='./Base/lists/lists_Synapse', help='list dir')
    parser.add_option('--root_path', type=str, default='./Dataset/Synapse/train_npz', help='root dir for data')
    parser.add_option('--train_path', type=str, default='D:\Program Files/vscode_code/myUnet1/Dataset/MoNuSeg/Train_Folder/', help='root dir for data')
    parser.add_option('--val_path', type=str, default='D:\Program Files/vscode_code/myUnet1/Dataset/MoNuSeg/Val_Folder/', help='root dir for data')

    parser.add_option('-o', '--log-path', type='str', dest='log_path', default='./log/', help='log path')
    parser.add_option('-m', type='str', dest='model', default='CoMsNet', help='use which model')
    parser.add_option('--num_class', type='int', dest='num_class', default=2, help='number of segmentation classes')
    parser.add_option('--base_chan', type='int', dest='base_chan', default=32,
                      help='number of channels of first expansion in UNet')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name', default='Comsnet_conv_bil',
                      help='unique experiment name')
    parser.add_option('--rlt', type='float', dest='rlt', default=1, help='relation between CE/FL and dice')
    parser.add_option('--weight', type='float', dest='weight',
                      default=[0.5, 1, 1, 1], help='weight each class in loss function')
    parser.add_option('--weight_decay', type='float', dest='weight_decay',
                      default=0.0001)
    parser.add_option('--scale', type='float', dest='scale', default=0.30)
    parser.add_option('--rotate', type='float', dest='rotate', default=180)
    parser.add_option('--crop_size', type='int', dest='crop_size', default=224)
    parser.add_option('--img_size', type='int', dest='img_size', default=224)

    parser.add_option('--aux_loss', dest='aux_loss', action='store_true', help='using aux loss for deep supervision')
    parser.add_option('--seed', type=int, default=1234, help='random seed')

    parser.add_option('--gpu', type='str', dest='gpu', default='0')
    options, args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = options.gpu

    print('Using model:', options.model)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # if options.model == 'UTNet':
    #     net = UTNet(1, options.base_chan, options.num_class, reduce_size=options.reduce_size,
    #                 block_list=options.block_list, num_blocks=options.num_blocks, num_heads=[4, 4, 4, 4],
    #                 projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=options.aux_loss,
    #                 maxpool=True)
    if options.model == 'CoMsNet':  # 默认224
        net = CoMsNet(3, options.base_chan, options.num_class, num_heads=[3, 6, 12, 24], num_block=0,
                      num_croblock=[0, 0, 0, 0],
                      attn_drop=0.1, maxpool=True)
    # SCUNet
    # if options.model == 'SCUNet':  # 默认256
    #     print("Scunet-----------start")
    #     net = SCUNet(in_nc=3, numclass=options.num_class)

    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))

    param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    # print(os.getcwd() + "====================")
    # print("sys.path----------------:",sys.path)
    # os.chdir("/home/lxt/exp_pre/pro_run/exp_run")
    net.to(device)
    # net.cuda()
    train_net(net, options,device)

    print('done')

    sys.exit(0)
