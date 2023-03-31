# encoding:utf-8

import os
import pdb
import torch
import random
import network
import argparse
import numpy as np
import os.path as osp
import torch.nn as nn
from datetime import date
import torch.optim as optim
from data_list import ImageList
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_tar_util import obtain_ncc_label
from dataset.data_transform import TransformSW
from utils import op_copy, lr_scheduler, image_train, image_test, cal_acc, print_args, log, set_log_path

# region constants

STD = [0.229, 0.224, 0.225]
MEAN = [0.485, 0.456, 0.406]
VISDA_CLASSES = ['aeroplane', 'bicycle', 'bus', 'car', 'horse' 'knife',
                 'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']

# endregion constants

# region functions


def trim_str(str: str, length: int) -> str:
    """
      Trim a string according to a given length
    """
    return str[:length] if len(str) >= length else str


def get_data_loader_dict(args) -> dict:
    """
      Load and prepare data based on the given args
    """
    # * read relevant args
    num_workers = args.worker
    data_trans_type = args.data_trans
    train_batch_size = args.batch_size
    src_dataset_path = args.src_dataset_path
    tar_dataset_path = args.tar_dataset_path
    test_dataset_path = args.test_dataset_path

    dataset_dict = {}
    dataset_loader_dict = {}
    txt_src = open(src_dataset_path, "r").readlines()
    txt_tar = open(tar_dataset_path, "r").readlines()
    txt_test = open(test_dataset_path, "r").readlines()

    # >>> source >>>

    dataset_size = len(txt_src)
    train_size = int(0.9 * dataset_size)
    # ! NOTE
    # * source validation set is a SUBSET of the source training set
    txt_src_train = txt_src
    _, txt_src_val = torch.utils.data.random_split(
        txt_src, [train_size, dataset_size - train_size])

    # * prepare the ImageList
    dataset_dict['src_val'] = ImageList(
        txt_src_val, transform=image_test(), root=os.path.dirname(src_dataset_path))
    dataset_dict['src_train'] = ImageList(
        txt_src_train, transform=image_train(), root=os.path.dirname(src_dataset_path))

    # * init the data loaders
    dataset_loader_dict['src_val'] = DataLoader(
        dataset_dict['src_val'], batch_size=train_batch_size, shuffle=True, num_workers=num_workers, drop_last=False)
    dataset_loader_dict['src_train'] = DataLoader(
        dataset_dict['src_train'], batch_size=train_batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

    # <<< source <<<

    # >>> target >>>

    data_transforms = TransformSW(
        MEAN, STD, arg_k=1) if data_trans_type == "SW" else image_train()
    dataset_dict['target'] = ImageList(
        txt_tar, transform=data_transforms, root=os.path.dirname(tar_dataset_path), ret_idx=True)
    dataset_loader_dict['target'] = DataLoader(
        dataset_dict['target'], batch_size=train_batch_size, shuffle=True, num_workers=num_workers, drop_last=False)

    # <<< target <<<

    # >>> test >>>

    dataset_dict['test'] = ImageList(
        txt_test, transform=image_test(), root=os.path.dirname(test_dataset_path), ret_idx=True)
    dataset_loader_dict['test'] = DataLoader(
        dataset_dict['test'], batch_size=train_batch_size * 3, shuffle=False, num_workers=num_workers, drop_last=False)

    # <<< test <<<

    return dataset_loader_dict


def run_adaptation(args, netF: nn.Module, netB: nn.Module, netC: nn.Module, dataset_loader_dict: dict) -> None:
    """
      Analyze the target based on the given args
    """
    # >>> retrieve relevant args >>>

    K = args.K
    KK = args.KK
    epsilon = args.epsilon
    max_epoch = args.max_epoch

    # <<< retrieve relevant args <<<

    dataset_loader_target = dataset_loader_dict["target"]

    # >>> optimizer >>>

    param_group = [{'params': netF.parameters(), 'lr': args.lr * 0.1},
                   {'params': netB.parameters(), 'lr': args.lr * 1}]
    param_group_c = [{'params': netC.parameters(), 'lr': args.lr * 1}]

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    # <<< optimizer <<<

    # >>> build feature and score bank >>>

    num_samples = len(dataset_loader_target.dataset)
    feat_bank = torch.randn(num_samples, 256)
    score_bank = torch.randn(num_samples, 12).cuda()

    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        data_iter = iter(dataset_loader_target)
        for _ in range(len(dataset_loader_target)):
            inputs, _, idx = data_iter.next()

            feat = netB(netF(inputs))
            feat_norm = F.normalize(feat)
            output = nn.Softmax(-1)(netC(feat))

            feat_bank[idx] = feat_norm.detach().clone().cpu()
            score_bank[idx] = output.detach().clone()

    # <<< build feature and score bank <<<

    # >>> start adaptation >>>
    netF.train()
    netB.train()
    netC.train()

    iter_count = 0
    max_iter = max_epoch * len(dataset_loader_target)

    while iter_count < max_iter:
        try:
            inputs, _, idx = data_iter.next()
        except:
            data_iter = iter(dataset_loader_target)
            inputs, _, idx = data_iter.next()

        if inputs.size(0) == 1:
            continue

        inputs = inputs.cuda()

        iter_count += 1
        lr_scheduler(optimizer, iter_num=iter_count, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_count, max_iter=max_iter)

        feat = netB(netF(inputs))
        output = netC(feat)
        softmax_output = nn.Softmax(dim=1)(output)

        with torch.no_grad():
            # * normalize and update feature/score bank
            feat_norm = F.normalize(feat).cpu().detach().clone()
            feat_bank[idx] = feat_norm.detach().clone().cpu()
            score_bank[idx] = softmax_output.detach().clone()

            distance = feat_norm@feat_bank.T
            _, idx_near = torch.topk(distance, dim=-1, largest=True, k=K+1)
            idx_near = idx_near[:, 1:]  # * leave out self-to-self distance
            feat_near = feat_bank[idx_near]  # * b x K
            score_near = score_bank[idx_near]  # * b x K x C

            feat_bank_re = feat_bank.unsqueeze(0).expand(
                feat_near.shape[0], -1, -1)  # * b x n x dim
            distance_re = torch.bmm(
                feat_near, feat_bank_re.permute(0, 2, 1))  # * b x K x n
            # * M nearest neighbors for each of above K ones
            _, idx_near_near = torch.topk(
                distance_re, dim=-1, largest=True, k=KK+1)
            # * leave out self-to-self distance
            idx_near_near = idx_near_near[:, :, 1:]
            idx_ = idx.unsqueeze(-1).unsqueeze(-1)
            match = (idx_near_near == idx_).sum(-1).float()  # * b x K
            weight = torch.where(match > 0., match, torch.ones_like(
                match).fill_(0.1))  # * b x K

            # * b x K x M
            weight_kk = weight.unsqueeze(-1).expand(-1, -1, KK).fill_(0.1)

            # * removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
            # * weight_kk[idx_near_near == tar_idx_]=0
            score_near_kk = score_bank[idx_near_near]  # * b x K x M x C
            weight_kk = weight_kk.contiguous().view(
                weight_kk.shape[0], -1)  # * b x KM

            score_near_kk = score_near_kk.contiguous().view(
                score_near_kk.shape[0], -1, 12)  # * b x KM x C

            score_self = score_bank[idx]

        # * nn of nn
        output_re = softmax_output.unsqueeze(
            1).expand(-1, K * KK, -1)  # * b x C x 1
        const = torch.mean(
            (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) * weight_kk.cuda()).sum(1))  # * kl_div here equals to dot product since we do not use log for score_near_kk
        loss = torch.mean(const)

        # * nn
        softmax_out_un = softmax_output.unsqueeze(
            1).expand(-1, K, -1)  # * b x K x C

        loss += torch.mean((
            F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.cuda()).sum(1))

        # * self, if not explicitly removing the self feature in expanded neighbor then no need for this
        # * loss += -torch.mean((softmax_out * score_self).sum(-1))

        msoftmax = softmax_output.mean(dim=0)
        gentropy_loss = torch.sum(
            msoftmax * torch.log(msoftmax + epsilon))
        loss += gentropy_loss

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()
    # <<< start adaptation <<<


def run_analysis(args, netF: nn.Module, netB: nn.Module, netC: nn.Module, dataset_loader_dict: dict) -> None:
    acc, cls_acc = cal_acc(
        dataset_loader_dict["target"], netF, netB, netC, flag=True)
    log("Source model accuracy on target domain after adaptation {} \n classwise accuracy {} \n".format(
        acc, [round(e, 3) for e in cls_acc]))

    pred, _ = obtain_ncc_label(
        dataset_loader_dict['target'], netF, netB, netC, args, log)
# endregion functions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare')

    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--net', type=str, default='resnet101')
    parser.add_argument('--dset', type=str, default='visda-2017')
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--seed', type=int, default=2021, help="random seed")
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--batch_size', type=int,
                        default=64, help="batch_size")
    parser.add_argument('--max_epoch', type=int,
                        default=15, help="max iterations")
    parser.add_argument('--worker', type=int, default=2,
                        help="number of workers")
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='0', help="device id to run")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--layer', type=str, default="wn",
                        choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str,
                        default="bn", choices=["ori", "bn"])
    parser.add_argument('--T', type=float, default=0.5,
                        help='Temperature for creating pseudo-label')
    parser.add_argument('--loss_type', type=str, default='sce',
                        help='Loss function for target domain adaptation')

    parser.add_argument('--distance', type=str,
                        default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--threshold', type=int, default=10,
                        help='threshold for filtering cluster centroid')

    parser.add_argument('--data_trans', type=str, default='W')
    parser.add_argument('--output', type=str, default='result/')
    parser.add_argument('--exp_name', type=str, default='Clust_BN')

    args = parser.parse_args()

    if args.dset == 'office-home':
        args.class_num = 65
        names = ['Art', 'Clipart', 'Product', 'RealWorld']

    elif args.dset == 'visda-2017':
        args.class_num = 12
        names = ['train', 'validation']

    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = '../dataset/'
        args.src_dataset_path = folder + args.dset + \
            '/' + names[args.s] + '/image_list.txt'
        args.tar_dataset_path = folder + args.dset + \
            '/' + names[args.t] + '/image_list.txt'
        args.test_dataset_path = folder + args.dset + \
            '/' + names[args.t] + '/image_list.txt'

        args.name = names[args.s][0].upper() + names[args.t][0].upper()
        args.output_dir_src = osp.join(
            args.output, args.dset, 'source', names[args.s][0].upper())
        args.output_dir = osp.join(
            args.output, args.dset, args.exp_name, names[args.s][0].upper() + names[args.t][0].upper())

        if not osp.exists(args.output_dir):
            os.makedirs(args.output_dir)
        set_log_path(args.output_dir)
        log('Log saved to {}'.format(args.output_dir_src))
        log(print_args(args))

        dataset_loader_dict = get_data_loader_dict(args)

        # * init base network
        netF = network.ResBase(res_name=args.net).cuda()
        netB = network.feat_bootleneck(
            type=args.classifier, feature_dim=netF.in_features, bottleneck_dim=args.bottleneck).cuda()
        netC = network.feat_classifier(
            type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()
        netF.load_state_dict(torch.load(f"{args.output_dir_src}/source_F.pt"))
        netB.load_state_dict(torch.load(f"{args.output_dir_src}/source_B.pt"))
        netC.load_state_dict(torch.load(f"{args.output_dir_src}/source_C.pt"))

        run_adaptation(args, netF, netB, netC, dataset_loader_dict)
        run_analysis(args, netF, netB, netC, dataset_loader_dict)
