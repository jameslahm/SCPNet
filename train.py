import os
import time
from typing import Tuple

import torch
from torch.cuda.amp import GradScaler, autocast  # type: ignore
from torch.optim import lr_scheduler

from log import logger
from loss import SPLC
from scpnet import SCPNetTrainer
from utils import AverageMeter, add_weight_decay, mAP

from config import cfg  # isort:skip

def save_best(trainer, if_ema_better: bool) -> None:
    if if_ema_better:
        torch.save(trainer.ema.module.state_dict(),
                    os.path.join(cfg.checkpoint, 'model-highest.ckpt'))
    else:
        torch.save(trainer.model.state_dict(),
                    os.path.join(cfg.checkpoint, 'model-highest.ckpt'))
    torch.save(trainer.model.state_dict(),
                os.path.join(cfg.checkpoint, 'model-highest-regular.ckpt'))
    torch.save(trainer.ema.module.state_dict(),
                os.path.join(cfg.checkpoint, 'model-highest-ema.ckpt'))

def validate(trainer, epoch: int) -> Tuple[float, bool]:

    trainer.model.eval()
    logger.info("Start validation...")
    sigmoid = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for _, (input, target) in enumerate(trainer.val_loader):
        target = target
        # compute output
        with torch.no_grad():
            with autocast():
                if cfg.model_name != 'simsiam':
                    output_regular = sigmoid(
                        trainer.model(input.cuda())).cpu()
                    output_ema = sigmoid(
                        trainer.ema.module(input.cuda())).cpu()
                else:
                    output_regular = sigmoid(
                        trainer.model.module.clip(
                            input.cuda())).cpu()
                    output_ema = sigmoid(
                        trainer.ema.module.module.clip(
                            input.cuda())).cpu()

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

    mAP_score_regular = mAP(
        torch.cat(targets).numpy(),
        torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(
        torch.cat(targets).numpy(),
        torch.cat(preds_ema).numpy())
    logger.info("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(
        mAP_score_regular, mAP_score_ema))
    mAP_max = max(mAP_score_regular, mAP_score_ema)
    if mAP_score_ema >= mAP_score_regular:
        if_ema_better = True
    else:
        if_ema_better = False

    trainer.model.train()
    return mAP_max, if_ema_better

def train(trainer) -> None:
    # set optimizer
    criterion = SPLC()
    parameters = add_weight_decay(trainer.model, cfg.weight_decay)
    max_lr = [cfg.lr, cfg.lr, cfg.gcn_lr, cfg.gcn_lr]
    optimizer = torch.optim.Adam(
        params=parameters, lr=cfg.lr,
        weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(trainer.train_loader)
    scheduler = lr_scheduler.OneCycleLR(  # type: ignore
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=cfg.total_epochs,  # type: ignore
        pct_start=0.2)

    highest_mAP = 0
    scaler = GradScaler()
    best_epoch = 0
    for epoch in range(cfg.epochs):
        for i, (input, target) in enumerate(trainer.train_loader):
            target = target.cuda()  # (batch,3,num_classes)
            # target = target.max(dim=1)[0]
            loss = trainer.train(input, target, criterion, epoch, i)

            trainer.model.zero_grad()
            scaler.scale(loss).backward()  # type: ignore
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            trainer.ema.update(trainer.model)
            if i % 100 == 0:
                logger.info('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                    .format(epoch, cfg.epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3), # noqa
                            scheduler.get_last_lr()[0], \
                            loss.item()))

        mAP_score, if_ema_better = validate(trainer, epoch)

        if mAP_score > highest_mAP:
            highest_mAP = mAP_score
            best_epoch = epoch
            save_best(trainer, if_ema_better)
        logger.info(
            'current_mAP = {:.2f}, highest_mAP = {:.2f}, best_epoch={}\n'.
            format(mAP_score, highest_mAP, best_epoch))
        logger.info("Save text embeddings done")

def test(trainer) -> None:
    # get model-highest.ckpt
    trainer.model.load_state_dict(
        torch.load(f"{cfg.checkpoint}/model-highest.ckpt"), strict=True)
    trainer.model.eval()

    logger.info("Start test...")
    batch_time = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()
    # mAP_meter = AverageMeter()

    sigmoid = torch.nn.Sigmoid()

    end = time.time()
    tp, fp, fn, tn, count = 0, 0, 0, 0, 0
    preds = []
    targets = []
    for i, (input, target) in enumerate(trainer.val_loader):
        target = target
        # compute output
        with torch.no_grad():
            output = sigmoid(trainer.model(input.cuda())).cpu()

        # for mAP calculation
        preds.append(output.cpu())
        targets.append(target.cpu())

        # measure accuracy and record loss
        pred = output.data.gt(cfg.thre).long()

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        # this_tn = (pred + target).eq(0).sum()

        this_prec = this_tp.float() / (this_tp + this_fp).float(
        ) * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (this_tp + this_fn).float(
        ) * 100.0 if this_tp + this_fn != 0 else 0.0

        prec.update(float(this_prec), input.size(0))
        rec.update(float(this_rec), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c = [
            float(tp[i].float() / (tp[i] + fp[i]).float()) *
            100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))
        ]
        r_c = [
            float(tp[i].float() / (tp[i] + fn[i]).float()) *
            100.0 if tp[i] > 0 else 0.0 for i in range(len(tp))
        ]
        f_c = [
            2 * p_c[i] * r_c[i] / (p_c[i] + r_c[i]) if tp[i] > 0 else 0.0
            for i in range(len(tp))
        ]

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        if i % 64 == 0:
            logger.info(
                'Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                    i,
                    len(trainer.val_loader),
                    batch_time=batch_time,
                    prec=prec,
                    rec=rec))
            logger.info(
                'P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    logger.info(
        '--------------------------------------------------------------------'
    )
    logger.info(
        ' * P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
        .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o,
                f_o))  # type: ignore

    mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    logger.info(f"mAP score: {mAP_score}")
    return torch.cat(targets).numpy(), torch.cat(preds).numpy()  # type: ignore



def main():
    trainer = SCPNetTrainer()
    if cfg.test:
        test(trainer)
    else:
        train(trainer)

if __name__ == '__main__':
    main()