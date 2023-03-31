# encoding:utf-8

import pdb
import numpy as np
import os.path as osp
from datetime import date
import argparse, os, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import network
from data_list import ImageList
from train_tar_util import obtain_ncc_label, bn_adapt
from loss import compute_dist, compute_loss
from dataset.data_transform import TransformSW
from utils import op_copy, lr_scheduler, image_train, image_test, cal_acc, print_args, log, set_log_path, pad_string


def trim_str(s, l):
    ret = s[:l] if len(s) >= l else s
    return ret


visda_classes = ['aeroplane', 'bicycle', 'bus', 'car', 'horse' 'knife', 'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9 * dsize)
    _, te_txt = torch.utils.data.random_split(txt_src, [tr_size, dsize - tr_size])
    tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train(), root=os.path.dirname(args.s_dset_path))
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test(), root=os.path.dirname(args.s_dset_path))
    dset_loaders["source_te"] = DataLoader(dsets["source_te"], batch_size=train_bs, shuffle=True,
                                           num_workers=args.worker, drop_last=False)

    data_trans = TransformSW(mean, std, aug_k=1) if args.data_trans == 'SW' else image_train()
    dsets["target"] = ImageList(txt_tar, transform=data_trans, root=os.path.dirname(args.t_dset_path), ret_idx=True)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList(txt_test, transform=image_test(), root=os.path.dirname(args.test_dset_path), ret_idx=True)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def analysis_target(args):
    dset_loaders = data_load(args)

    ## set base network
    netF = network.ResBase(res_name=args.net).cuda()
    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))

    param_group = [{'params': netF.parameters(), 'lr': args.lr * 0.1},
                   {'params': netB.parameters(), 'lr': args.lr * 1}]
    param_group_c = [{'params': netC.parameters(), 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    # performance of original model
    acc, cls_acc = cal_acc(dset_loaders["target"], netF, netB, netC, flag=True)
    log("Source model accuracy on target domain {} \n classwise accuracy {} \n".format(acc, [round(e, 3) for e in cls_acc]))

    pred, _ = obtain_ncc_label(dset_loaders["target"], netF, netB, netC, args, log)

    log("Adapt Batch Norm parameters")
    netF, netB = bn_adapt(netF, netB, dset_loaders["target"], runs=1000)

    pred, _ = obtain_ncc_label(dset_loaders["target"], netF, netB, netC, args, log)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neighbors')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=2, help="number of workers")
    parser.add_argument('--dset', type=str, default='visda-2017')
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101')
    parser.add_argument('--seed', type=int, default=2021, help="random seed")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--T', type=float, default=0.5, help='Temperature for creating pseudo-label')
    parser.add_argument('--loss_type', type=str, default='sce', help='Loss function for target domain adaptation')

    parser.add_argument('--distance', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--threshold', type=int, default=10, help='threshold for filtering cluster centroid')

    parser.add_argument('--output', type=str, default='result/')
    parser.add_argument('--exp_name', type=str, default='Clust_BN')
    parser.add_argument('--data_trans', type=str, default='W')
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = '../dataset/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '/image_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '/image_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '/image_list.txt'

        args.output_dir_src = osp.join(args.output, args.dset, 'source', names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.dset, args.exp_name, names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.makedirs(args.output_dir)
        set_log_path(args.output_dir)
        log('save log to path {}'.format(args.output_dir_src))
        log(print_args(args))

        analysis_target(args)
