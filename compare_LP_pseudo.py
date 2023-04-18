# encoding:utf-8

import os
import time
import scipy
import torch
import faiss
import random
import network
import argparse
import tracemalloc
import numpy as np
import os.path as osp
import torch.nn as nn
from typing import List
from datetime import date
import torch.optim as optim
from faiss import normalize_L2
from data_list import ImageList
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist
from train_tar_util import obtain_ncc_label
from sklearn.metrics import confusion_matrix
from dataset.data_transform import TransformSW
from utils import op_copy, lr_scheduler, image_train, image_test, cal_acc, print_args, log, set_log_path, Entropy

# region constants

STD = [0.229, 0.224, 0.225]
MEAN = [0.485, 0.456, 0.406]
VISDA_CLASSES = ['aeroplane', 'bicycle', 'bus', 'car', 'horse' 'knife',
                 'motorcycle', 'person', 'plant', 'skateboard', 'train', 'truck']

# endregion constants

# region class


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * float(n)
        self.count += float(n)
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.val:{format}} ({self.avg:{format}})".format(self=self, format=format)


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if not name in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=''):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix='/avg'):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix='/sum'):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix='/count'):
        return {name + postfix: meter.count for name, meter in self.meters.items()}
# endregion class

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


def run_ncc(args, loader: DataLoader, netF: nn.Module, netB: nn.Module, netC: nn.Module, log_func=print):
    """
        prediction from Nearest Centroid Classifier
    """
    # * read relevant args
    distance = args.distance
    threshold = args.threshold

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            input, label = data[0], data[1]
            feat = netB(netF(input.cuda()))
            output = netC(feat)
            if start_test:
                all_feat = feat.float().cpu()
                all_output = output.float().cpu()
                all_label = label.float()
                start_test = False
            else:
                all_feat = torch.cat((all_feat, feat.float().cpu()), 0)
                all_output = torch.cat((all_output, output.float().cpu()), 0)
                all_label = torch.cat((all_label, label.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    #ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    #unknown_weight = 1 - ent / np.log(args.class_num)
    _, predict = torch.max(all_output, 1)

    acc = torch.sum(torch.squeeze(predict).float() ==
                    all_label).item() / float(all_label.size()[0])
    if distance == 'cosine':
        all_feat = torch.cat((all_feat, torch.ones(all_feat.size(0), 1)), 1)
        all_feat = (all_feat.t() / torch.norm(all_feat, p=2, dim=1)).t()

    all_feat = all_feat.float().cpu().numpy()  # [B, 257]
    aff = all_output.float().cpu().numpy()     # [B, 12]

    for _ in range(2):
        initc = aff.transpose().dot(all_feat)  # [12, B] [B, 257]
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        cls_count = np.eye(all_output.size(1))[predict].sum(axis=0)
        labelset = np.where(cls_count > threshold)[0]

        dd = cdist(all_feat, initc[labelset], distance)
        pred_label = dd.argmin(axis=1)
        new_pred = labelset[pred_label]
        pred_score = torch.zeros_like(all_output).float()
        pred_score[:, labelset] = nn.Softmax(dim=1)(torch.tensor(dd)).float()

    new_acc = np.sum(new_pred == all_label.float().numpy()) / len(all_feat)
    log_func('Nearest Clustering Centroid Based Accuracy = {:.2f}% -> {:.2f}%'.format(
        acc * 100, new_acc * 100))
    return new_pred.astype('int'), all_feat, all_label, pred_score


def run_pseudo(loader: DataLoader, netF: nn.Module, netB: nn.Module, netC: nn.Module, log_func=print) -> List[torch.Tensor]:

    start_test = True
    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat(
                    (all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    # breakpoint()

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() ==
                         all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(all_output)).cpu().data.item()

    log_func(
        f"Pseudo -> accuracy: {accuracy*100}; mean entropy: {mean_ent}")


def run_label_propagation(args, loader: DataLoader, feat: torch.Tensor,
                          netF: nn.Module, netB: nn.Module, netC: nn.Module,
                          label: torch.Tensor, pred_label: torch.Tensor, log_func=print) -> None:
    # * get relevant args
    output_dir_src = args.output_dir_src

    for k in np.arange(3, 11):
        args.k = k
        netF.load_state_dict(torch.load(f"{output_dir_src}/source_F.pt"))
        netB.load_state_dict(torch.load(f"{output_dir_src}/source_B.pt"))
        netC.load_state_dict(torch.load(f"{output_dir_src}/source_C.pt"))
        # * obtain pseudo labels via label propagation
        weights, class_weights, p_labels = _update_plabels(
            args, feat, pred_label, log_func=log_func)
        new_acc = np.sum(p_labels == np.asarray(label)) / len(label)
        log('accuracy after label propagation with k={} is {:.4f}'.format(k, new_acc))

        _train_with_plabels(loader, netF, netB, netC, weights,
                            class_weights, p_labels=p_labels, log_func=log_func)

        start_test = True
        netF.eval()
        netB.eval()
        netC.eval()
        with torch.no_grad():
            iter_test = iter(loader)
            for i in range(len(loader)):
                data = next(iter_test)
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                outputs = netC(netB(netF(inputs)))
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat(
                        (all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)

        all_output = nn.Softmax(dim=1)(all_output)
        _, predict = torch.max(all_output, 1)
        accuracy = torch.sum(torch.squeeze(predict).float() ==
                             all_label).item() / float(all_label.size()[0])
        mean_ent = torch.mean(Entropy(all_output)).cpu().data.item()

        log_func(
            f"LP -> accuracy: {accuracy*100}; mean entropy: {mean_ent}; k: {args.k}")


def _update_plabels(args, feat: torch.Tensor, pred_label: torch.Tensor, log_func=print, alpha: float = 0.99, max_iter: int = 20) -> torch.Tensor:
    # * read relevant args
    k = args.k
    class_num = args.class_num

    log_func('======= Updating pseudo-labels =======')
    pred_label = np.asarray(pred_label)

    # * kNN search for the graph
    N, d = feat.shape[0], feat.shape[1]
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    # * build the index
    index = faiss.GpuIndexFlatIP(res, d, flat_config)

    normalize_L2(feat)
    index.add(feat)
    log(f"n total {index.ntotal}")
    D, I = index.search(feat, k + 1)

    # * create the graph
    # * [N, k]
    D = D[:, 1:] ** 3
    I = I[:, 1:]
    row_idx = np.arange(N)
    row_idx_rep = np.tile(row_idx, (k, 1)).T
    W = scipy.sparse.csr_matrix(
        (D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
    W = W + W.T

    # * Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axis=1)
    S[S == 0] = 1
    D = np.array(1. / np.sqrt(S))
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    # * Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
    Z = np.zeros((N, int(class_num)))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn
    for i in range(int(class_num)):
        cur_idx = np.where(pred_label == i)[0]
        y = np.zeros((N,))
        y[cur_idx] = 1.0 / cur_idx.shape[0]
        f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=max_iter)
        Z[:, i] = f

    # * Handle numerical errors
    Z[Z < 0] = 0

    # * Compute the weight for each instance based on the entropy (eq 11 from the paper)
    probs_l1 = F.normalize(torch.tensor(Z), p=1, dim=1).numpy()
    probs_l1[probs_l1 < 0] = 0
    entropy = scipy.stats.entropy(probs_l1.T)
    weights = 1 - entropy / np.log(int(class_num))
    weights = weights / np.max(weights)
    p_labels = np.argmax(probs_l1, 1)

    class_weights = np.ones((int(class_num),), dtype=np.float32)
    # * Compute the weight for each class
    for i in range(int(class_num)):
        cur_idx = np.where(np.asarray(p_labels) == i)[0]
        class_weights[i] = (float(pred_label.shape[0]) /
                            int(class_num)) / cur_idx.size

    return weights, class_weights, p_labels


def _train_with_plabels(loader: DataLoader, netF: nn.Module, netB: nn.Module, netC: nn.Module, weights, class_weights, p_labels, max_iter=1, log_func=print):
    # * loss functions
    class_criterion = nn.CrossEntropyLoss(
        ignore_index=-1, reduction="none").cuda()

    meters = AverageMeterSet()

    start_test = True
    iter_test = iter(loader)
    for i in range(len(loader)):
        data = next(iter_test)
        inputs = data[0]
        labels = data[1]
        inputs = inputs.cuda()

        if start_test:
            all_inputs = inputs
            all_label = labels
            start_test = False
        else:
            all_inputs = torch.cat(
                (all_inputs, inputs), 0)
            all_label = torch.cat((all_label, labels), 0)

    # * put all_inputs and all_labels to GPU
    all_inputs = all_inputs.cuda()
    all_labels = all_labels.float()

    # * init the optimizers
    param_group = [{'params': netF.parameters(), 'lr': args.lr * 0.1},
                   {'params': netB.parameters(), 'lr': args.lr * 1}]
    param_group_c = [{'params': netC.parameters(), 'lr': args.lr * 1}]

    optimizer = op_copy(optim.SGD(param_group))
    optimizer_c = op_copy(optim.SGD(param_group_c))

    netF.train()
    netB.train()
    netC.train()

    end = time.time()
    for _ in range(max_iter):
        meters.update(f"Data time: {time.time() - end}")

        input_var = torch.autograd.Variable(all_inputs)
        target_var = torch.autograd.Variable(p_labels)
        weights_var = torch.autograd.Variable(weights)
        class_weights_var = torch.autograd.Variable(class_weights)

        b_size = len(target_var)
        meters.update(f"Mini-batch size: {b_size}")

        class_logits = netC(netB(netF(input_var)))

        # * compute the loss
        loss = class_criterion(class_logits, target_var)
        loss = loss * weights_var.float()
        loss = loss * class_weights_var
        loss = loss.sum() / b_size
        meters.update("class_loss", loss.item())

        # * back-propagation
        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()

        meters.update(f"Batch time: {time.time() - end}")
        end = time.time()

    netF.eval()
    netB.eval()
    netC.eval()
    with torch.no_grad():
        all_output = netC(netB(netF(all_inputs)))

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() ==
                         all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(Entropy(all_output)).cpu().data.item()

    log_func(
        f"After LP -> accuracy: {accuracy*100}; mean entropy: {mean_ent}")

# endregion functions


if __name__ == "__main__":
    # * argument parsing
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
    parser.add_argument('--exp_name', type=str, default='LP_ncc_pseudo')

    parser.add_argument('--k', type=int, default=10,
                        help='number of neighbors for label propagation')

    args = parser.parse_args()

    # * dataset preliminaries
    if args.dset == 'office-home':
        args.class_num = 65
        names = ['Art', 'Clipart', 'Product', 'RealWorld']

    elif args.dset == 'visda-2017':
        args.class_num = 12
        names = ['train', 'validation']

    # * seeds and env variables
    SEED = args.seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # * iterate over source-target combinations
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

        # breakpoint()
        # * run analysis
        run_pseudo(dataset_loader_dict['target'],
                   netF, netB, netC, log_func=log)
        # breakpoint()
        pred_label, feat, label, _ = run_ncc(
            args, dataset_loader_dict['target'], netF, netB, netC, log_func=log)
        # breakpoint()
        run_label_propagation(
            args, dataset_loader_dict['target'], feat, netF, netB, netC, label, pred_label, log_func=log)
