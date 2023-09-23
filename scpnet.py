import copy
import os

import numpy as np
import torch
from PIL import Image
from torch.cuda.amp import autocast  # type: ignore
from torchvision import transforms

from config import cfg
from log import logger
from model import SCPNet, load_clip_model
from utils import COCO_missing_val_dataset, CocoDetection, ModelEma, get_ema_co

from randaugment import RandAugment


class WeakStrongDataset(torch.utils.data.Dataset):  # type: ignore

    def __init__(self,
                 root,
                 annFile,
                 transform,
                 target_transform=None,
                 class_num: int = -1):
        self.root = root
        with open(annFile, 'r') as f:
            names = f.readlines()
        self.name = names
        self.transform = transform
        self.class_num = class_num
        self.target_transform = target_transform
        self.strong_transform: transforms.Compose = copy.deepcopy(
            transform)  # type: ignore
        self.strong_transform.transforms.insert(0,
                                                RandAugment(3,
                                                            5))  # type: ignore

    def __getitem__(self, index):
        name = self.name[index]
        path = name.strip('\n').split(',')[0]
        num = name.strip('\n').split(',')[1]
        num = num.strip(' ').split(' ')
        num = np.array([int(i) for i in num])
        label = np.zeros([self.class_num])
        label[num] = 1
        label = torch.tensor(label, dtype=torch.long)
        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        img_w = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)  # type: ignore # noqa
        assert (self.target_transform is None)
        return [index, img_w,
                self.transform(img),
                self.strong_transform(img)], label

    def __len__(self):
        return len(self.name)


def build_weak_strong_dataset(train_preprocess,
                              val_preprocess,
                              pin_memory=True):
    if "coco" in cfg.data:
        return build_coco_weak_strong_dataset(train_preprocess, val_preprocess)
    elif "nuswide" in cfg.data:
        return build_nuswide_weak_strong_dataset(train_preprocess,
                                                 val_preprocess)
    elif "voc" in cfg.data:
        return build_voc_weak_strong_dataset(train_preprocess, val_preprocess)
    elif "cub" in cfg.data:
        return build_cub_weak_strong_dataset(train_preprocess, val_preprocess)
    else:
        assert (False)


def build_coco_weak_strong_dataset(train_preprocess, val_preprocess):

    # COCO Data loading
    instances_path_val = os.path.join(cfg.data,
                                      'annotations/instances_val2014.json')
    # instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    instances_path_train = cfg.dataset

    data_path_val = f'{cfg.data}/val2014'  # args.data
    data_path_train = f'{cfg.data}/train2014'  # args.data
    val_dataset = CocoDetection(data_path_val, instances_path_val,
                                val_preprocess)
    train_dataset = WeakStrongDataset(data_path_train,
                                      instances_path_train,
                                      train_preprocess,
                                      class_num=cfg.num_classes)

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(  # type: ignore
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.workers,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(  # type: ignore
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=False)

    return [train_loader, val_loader]


def build_nuswide_weak_strong_dataset(train_preprocess, val_preprocess):
    # Nus_wide Data loading
    instances_path_train = cfg.train_dataset
    instances_path_val = cfg.val_dataset

    data_path_val = f'{cfg.data}images'  # args.data
    data_path_train = f'{cfg.data}images'  # args.data

    val_dataset = COCO_missing_val_dataset(data_path_val,
                                       instances_path_val,
                                       val_preprocess,
                                       class_num=cfg.num_classes)
    train_dataset = WeakStrongDataset(data_path_train,
                                      instances_path_train,
                                      train_preprocess,
                                      class_num=cfg.num_classes)
    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.workers,
                                             pin_memory=False)
    return [train_loader, val_loader]


def build_voc_weak_strong_dataset(train_preprocess, val_preprocess):
    # VOC Data loading
    instances_path_train = cfg.train_dataset
    instances_path_val = cfg.val_dataset

    data_path_val = f'{cfg.data}VOC2012/JPEGImages'  # args.data
    data_path_train = f'{cfg.data}VOC2012/JPEGImages'  # args.data

    val_dataset = COCO_missing_val_dataset(data_path_val,
                                       instances_path_val,
                                       val_preprocess,
                                       class_num=cfg.num_classes)
    train_dataset = WeakStrongDataset(data_path_train,
                                      instances_path_train,
                                      train_preprocess,
                                      class_num=cfg.num_classes)
    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.workers,
                                             pin_memory=False)
    return [train_loader, val_loader]


def build_cub_weak_strong_dataset(train_preprocess, val_preprocess):
    # CUB Data loading
    instances_path_train = cfg.train_dataset
    instances_path_val = cfg.val_dataset

    data_path_val = f'{cfg.data}CUB_200_2011/images'  # args.data
    data_path_train = f'{cfg.data}CUB_200_2011/images'  # args.data

    val_dataset = COCO_missing_val_dataset(data_path_val,
                                       instances_path_val,
                                       val_preprocess,
                                       class_num=cfg.num_classes)
    train_dataset = WeakStrongDataset(data_path_train,
                                      instances_path_train,
                                      train_preprocess,
                                      class_num=cfg.num_classes)
    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size,
                                               shuffle=True,
                                               num_workers=cfg.workers,
                                               pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=cfg.batch_size,
                                             shuffle=False,
                                             num_workers=cfg.workers,
                                             pin_memory=False)
    return [train_loader, val_loader]

class SCPNetTrainer():

    def __init__(self) -> None:
        super().__init__()

        clip_model, _ = load_clip_model()
        # image_size = clip_model.visual.input_resolution
        image_size = cfg.image_size

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                         (0.26862954, 0.26130258, 0.27577711))

        train_preprocess = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor(), normalize
        ])
        val_preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(), normalize
        ])

        train_loader, val_loader = build_weak_strong_dataset(
            train_preprocess,  # type: ignore
            val_preprocess)
        self.train_loader = train_loader
        self.val_loader = val_loader

        classnames = val_loader.dataset.labels()
        assert (len(classnames) == cfg.num_classes)

        self.model = SCPNet(classnames, clip_model)
        self.relation = self.model.relation
        self.classnames = classnames
        for name, param in self.model.named_parameters():
            if "text_encoder" in name:
                param.requires_grad_(False)

        self.model.cuda()
        ema_co = get_ema_co()
        self.ema = ModelEma(self.model, ema_co)  # 0.9997^641=0.82

        self.selected_label = torch.zeros(
            (len(self.train_loader.dataset), cfg.num_classes),
            dtype=torch.long,
        )
        self.selected_label = self.selected_label.cuda()
        self.classwise_acc = torch.zeros((cfg.num_classes, )).cuda()
        self.classwise_acc[:] = 1/cfg.num_classes

    def consistency_loss(self, logits_s, logits_w, y_lb):
        logits_w = logits_w.detach()

        pseudo_label = torch.sigmoid(logits_w)
        pseudo_label_s = torch.sigmoid(logits_s)

        relation_p = pseudo_label @ self.relation.cuda().t()

        max_probs, max_idx = torch.topk(pseudo_label, cfg.hard_k, dim=-1)
        threhold = cfg.p_cutoff * (self.classwise_acc[max_idx] /
                                    (2. - self.classwise_acc[max_idx]))
        mask = max_probs.ge(threhold).float().sum(dim=1) >= 1  # convex
        labels = torch.zeros((len(logits_s), cfg.num_classes),
                                dtype=torch.long)
        for i, idx in enumerate(max_idx):
            labels[i][idx] = 1
        labels_mask = pseudo_label < cfg.p_cutoff * (
            self.classwise_acc / (2. - self.classwise_acc))
        labels[labels_mask] = 0
        labels = torch.logical_or(labels, y_lb.cpu()).type(torch.long)
        labels = labels.cuda()
        xs_pos = pseudo_label_s
        xs_neg = 1 - pseudo_label_s
        los_pos = labels * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - labels) * torch.log(xs_neg.clamp(min=1e-8))
        loss = (los_pos + los_neg) * mask.reshape(-1, 1)
        loss_kl = (relation_p * torch.log(xs_pos.clamp(min=1e-8)) + (1 - relation_p) * torch.log(xs_neg.clamp(min=1e-8))) * mask.reshape(-1, 1)
        return -loss.sum() - cfg.kl_lambda * loss_kl.sum(), labels

    def train(self, input, target, criterion, epoch, epoch_i) -> torch.Tensor:
        
        x_ulb_idx, x_lb, x_ulb_w, x_ulb_s = input
        y_lb = target

        num_lb = x_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]
        assert num_ulb == x_ulb_s.shape[0]

        x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(), x_ulb_w.cuda(), x_ulb_s.cuda()
        x_ulb_idx = x_ulb_idx.cuda()

        pseudo_counter = self.selected_label.sum(dim=0)
        max_v = pseudo_counter.max().item()
        sum_v = pseudo_counter.sum().item()
        if max_v >= 1:  # not all(5w) -1
            for i in range(cfg.num_classes):
                self.classwise_acc[i] = max(pseudo_counter[i] / max(
                    max_v,
                    cfg.hard_k * len(self.selected_label) - sum_v), 1/cfg.num_classes)

        inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))

        # inference and calculate sup/unsup losses
        with autocast():
            logits = self.model(inputs)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)
        logits_x_lb = logits_x_lb.float()
        logits_x_ulb_w, logits_x_ulb_s = logits_x_ulb_w.float(
        ), logits_x_ulb_s.float()

        sup_loss, _ = criterion(logits_x_lb, y_lb, epoch)

        unsup_loss, labels = self.consistency_loss(logits_x_ulb_s,
                                                   logits_x_ulb_w, y_lb)

        assert (labels is not None)
        select_mask = labels.sum(dim=1) >= 1
        if x_ulb_idx[select_mask].nelement() != 0:
            self.selected_label[
                x_ulb_idx[select_mask]] = labels[select_mask]

        total_loss = sup_loss + cfg.lambda_u * unsup_loss

        return total_loss
