import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from Base.datasets.dataset_synapse import Synapse_dataset, RandomGenerator
from UT_base.utnet import UTNet, UTNet_Encoderonly

from UT_tr.losses import DiceLoss2, test_single_volume
from UT_tr.utils.utils import *
from UT_tr.utils import metrics
from optparse import OptionParser
import SimpleITK as sitk

from torch.utils.tensorboard import SummaryWriter
import time
import math
import os
import sys
import pdb
import warnings
import logging

from model.Coms_decoADj import CoMs_deAdj
from model.coms_At import CoMs_AT
warnings.filterwarnings("ignore", category=UserWarning)
DEBUG = False

def train_net(net, options):

    logging.basicConfig(filename=options.cp_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(options))
    
    data_path = options.root_path

    db_train = Synapse_dataset(base_dir=data_path, list_dir=options.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[options.crop_size, options.crop_size])]))

    print("The length of train set is: {}".format(len(db_train)))

    def worker_init_fn(worker_id):
        torch.random.seed(options.seed + worker_id)

    # batch_size
    trainloader = DataLoader(db_train, batch_size=options.batch_size, shuffle=True, num_workers=0, pin_memory=True,
                             worker_init_fn=worker_init_fn)


    writer = SummaryWriter(options.log_path + options.unique_name)

    optimizer = optim.SGD(net.parameters(), lr=options.lr, momentum=0.9, weight_decay=options.weight_decay)

    ce_loss = CrossEntropyLoss()
    cls_num = 9
    dice_loss = DiceLoss2(cls_num)
    iter_num = 0
    base_lr = options.lr

    max_iterations = options.epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    best_dice = 0.15
    for epoch in range(options.epochs):

        print('Starting epoch {}/{}'.format(epoch+1, options.epochs))
        epoch_loss = 0

        # exp_scheduler = exp_lr_scheduler_with_warmup(optimizer, init_lr=options.lr, epoch=epoch, warmup_epoch=5, max_epoch=options.epochs)
        #
        print('current lr:', base_lr)


        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # img : torch.Size([6, 1, 224, 224]) label : torch.Size([6, 224, 224])
            image_batch, label_batch = image_batch, label_batch
            # output torch.Size([6, 9, 224, 224])
            # image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            image_batch, label_batch = image_batch, label_batch

            end = time.time()
            net.train()
            with torch.autograd.set_detect_anomaly(True):
                outputs = net(image_batch)
                loss_ce = ce_loss(outputs, label_batch[:].long())
                loss_dice = dice_loss(outputs, label_batch, softmax=True)
                loss = 0.5 * loss_ce + 0.5 * loss_dice

                lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

                iter_num = iter_num + 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            batch_time = time.time() - end
            print('batch loss: %.5f, batch_time:%.5f'%(loss.item(), batch_time))
        print('[epoch %d] epoch loss: %.5f'%(epoch+1, epoch_loss/(i_batch+1)))
        loss_cn = epoch_loss/(i_batch+1)

        writer.add_scalar('Train/Loss', epoch_loss/(i_batch+1), epoch+1)
        writer.add_scalar('LR', lr_, epoch+1)

        if os.path.isdir('%s%s/'%(options.cp_path, options.unique_name)):
            pass
        else:
            os.mkdir('%s%s/'%(options.cp_path, options.unique_name))

        # if epoch % 20 == 0 or epoch > options.epochs-10:
        #     torch.save(net.state_dict(), '%s%s/CP%d.pth'%(options.cp_path, options.unique_name, epoch))
        #     save_mode_path = os.path.join(options.cp_path, options.unique_name, epoch)
        #     logging.info("save model to {}".format(save_mode_path))
        if loss_cn <= best_dice:
            best_dice = loss_cn
            # if epoch >= options.epochs - 1:
            torch.save(net.state_dict(), '%s%s/%d_sc%d__best.pth' % (options.cp_path, options.unique_name, str(epoch+1), str(best_dice)))
            save_mode_path = os.path.join(options.cp_path, options.unique_name,  str(epoch+1), str(best_dice))
            logging.info("save model to {}".format(save_mode_path))









if __name__ == '__main__':
    parser = OptionParser()
    def get_comma_separated_int_args(option, opt, value, parser):
        value_list = value.split(',')
        value_list = [int(i) for i in value_list]
        setattr(parser.values, option.dest, value_list)

    parser.add_option('--is_savenii', action="store_true", help='whether to save results during inference')
    parser.add_option('--volume_path', type=str, default='../Dataset/Synapse/test_vol_h5', help='root dir for validation volume data')  # for acdc volume_path=root_dir)
    # 默认迭代150次
    parser.add_option('-e', '--epochs', dest='epochs', default=150, type='int', help='number of epochs')
    # 默认batch_size = 24
    parser.add_option('-b', '--batch_size', dest='batch_size', default=6, type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01, type='float', help='learning rate')
    parser.add_option('-c', '--resume', type='str', dest='load', default=False, help='load pretrained model')
    parser.add_option('-p', '--checkpoint-path', type='str', dest='cp_path', default='./checkpoint/', help='checkpoint path')
    parser.add_option('--list_dir', type=str, default='../Base/lists/lists_Synapse', help='list dir')
    parser.add_option('--root_path', type=str, default='../Dataset/Synapse/train_npz', help='root dir for data')

    parser.add_option('-o', '--log-path', type='str', dest='log_path', default='./log/', help='log path')
    parser.add_option('-m', type='str', dest='model', default='CoMs_AT', help='use which model')
    parser.add_option('--num_class', type='int', dest='num_class', default=9, help='number of segmentation classes')
    parser.add_option('--base_chan', type='int', dest='base_chan', default=32, help='number of channels of first expansion in UNet')
    parser.add_option('-u', '--unique_name', type='str', dest='unique_name', default='CoMs_deAdj', help='unique experiment name')
    parser.add_option('--rlt', type='float', dest='rlt', default=1, help='relation between CE/FL and dice')
    parser.add_option('--weight', type='float', dest='weight',
                      default=[0.5,1,1,1] , help='weight each class in loss function')
    parser.add_option('--weight_decay', type='float', dest='weight_decay',
                      default=0.0001)
    parser.add_option('--scale', type='float', dest='scale', default=0.30)
    parser.add_option('--rotate', type='float', dest='rotate', default=180)
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

    if options.model == 'UTNet':
        net = UTNet(1, options.base_chan, options.num_class, reduce_size=options.reduce_size, block_list=options.block_list, num_blocks=options.num_blocks, num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=options.aux_loss, maxpool=True)
    if options.model == 'CoMs_deAdj':
        net = CoMs_deAdj(3, options.base_chan, options.num_class, num_heads=[3, 6, 12, 24], num_block=0, num_croblock=[0, 0, 0, 0],
                        attn_drop=0.1, maxpool=True)
    if options.model == 'CoMs_AT':
        net = CoMs_AT(3, 32, 9, num_heads=[3, 6, 12, 24], num_block=0, num_croblock=[0, 0, 0, 0],
                  attn_drop=0.1, maxpool=True)

    if options.load:
        net.load_state_dict(torch.load(options.load))
        print('Model loaded from {}'.format(options.load))
    
    # param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    #
    # print(net)
    # print(param_num)
    
    # net.cuda()
    
    # train_net(net, options)

    from ptflops import get_model_complexity_info

    # with torch.cuda.device(0):

    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print('done')

    sys.exit(0)
