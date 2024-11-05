import argparse
import os, sys
import time
import random
import json
import numpy as np

import torch
from torch.optim import lr_scheduler
import torch.optim
import torch.utils.data

from torch.utils.tensorboard import SummaryWriter

import _init_paths
from dataset.get_dataset import get_datasets

from models.Resnet import create_model

import matplotlib.pyplot as plt

from utils.logger import setup_logger
from utils.meter import AverageMeter, AverageMeterHMS, ProgressMeter
from utils.helper import clean_state_dict, function_mAP, get_raw_dict, ModelEma, add_weight_decay
from utils.losses import AsymmetricLoss

NUM_CLASS = {'voc': 20, 'coco': 80, 'nus': 81}

def parser_args():
    parser = argparse.ArgumentParser(description='Main')

    # data
    parser.add_argument('--dataset_name', default='coco', choices=['voc', 'coco', 'nus'], 
                        help='dataset name')
    parser.add_argument('--dataset_dir',  default='./data', metavar='DIR', 
                        help='dir of all datasets')
    parser.add_argument('--img_size', default=224, type=int,
                        help='size of input images')
    parser.add_argument('--output', default='./outputs', metavar='DIR', 
                        help='path to output folder')
    parser.add_argument('--lb_ratio', default=0.05, type=float,
                        help='the ratio of lb:(lb+ub)')
    parser.add_argument('--cutout', default=0.5, type=float, 
                        help='cutout factor')
    parser.add_argument("--grid_perside", type=int, default=2,
                        help="Number of grids per side for local images. (example: 2)")

    # train
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--warmup_batch_size', default=32, type=int,
                        help='batch size for warmup')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', 
                        help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,metavar='W', 
                        help='weight decay (default: 1e-2)', dest='weight_decay')
    parser.add_argument('-p', '--print_freq', default=400, type=int, metavar='N', 
                        help='print frequency (default: 10)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='apply amp')
    parser.add_argument('--early_stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--optim', default='adamw', type=str,
                        help='optimizer used')
    parser.add_argument('--warmup_epochs', default=12, type=int,
                        help='the number of epochs for warmup')
    parser.add_argument('--loss_lb', default='asl', type=str, 
                        help='used_loss for lb')
    parser.add_argument('--loss_ub', default='asl', type=str, 
                        help='used_loss for ub')
    parser.add_argument('--cat_strategy', default='f1', type=str, 
                        help='')
    parser.add_argument('--beta', default=1.0, type=float, 
                        help='')


    # random seed
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')

    # model
    parser.add_argument('--net', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--is_data_parallel', action='store_true', default=False,
                        help='on/off nn.DataParallel()')
    parser.add_argument("--parallel_devices", type=int, default=[0,1], nargs="+",
                        help="device list (example: [0,1])")
    parser.add_argument('--ema_decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--resume', default=None, type=str, 
                        help='path to latest checkpoint (default: none)')


    args = parser.parse_args()
    
    if args.lb_ratio <= 0.05:
        args.lb_bs = 4
        args.ub_bs = 60
    elif args.lb_ratio == 0.1:
        args.lb_bs = 8
        args.ub_bs = 56
    elif args.lb_ratio == 0.15:
        args.lb_bs = 12
        args.ub_bs = 52
    elif args.lb_ratio == 0.2:
        args.lb_bs = 16
        args.ub_bs = 48

    if args.dataset_name == 'voc':
        args.lb_bs = int(args.lb_bs / 2)
        args.ub_bs = int(args.ub_bs / 2)

    args.output = args.net + '_outputs'
    args.resume = '%s_outputs/%s/%s/warmup_%s_%s/warmup_model.pth.tar'%(args.net, args.dataset_name, args.lb_ratio, args.loss_lb, args.warmup_epochs)
    args.n_classes = NUM_CLASS[args.dataset_name]
    args.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name) 
    
    args.output = os.path.join(args.output, args.dataset_name,  '%s'%args.lb_ratio,  'ours_%s_%s_%s_%s_%s_%s'%(args.cat_strategy, args.warmup_epochs, args.cutout, args.lr, args.lb_bs, args.beta))

    return args


def get_args():
    args = parser_args()
    return args


def main():
    args = get_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    os.makedirs(args.output, exist_ok=True)

    logger = setup_logger(output=args.output, color=False, name="XXX")
    logger.info("Command: "+' '.join(sys.argv))

    path = os.path.join(args.output, "config.json")
    with open(path, 'w') as f:
        json.dump(get_raw_dict(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))

    return main_worker(args, logger)

def main_worker(args, logger):
    # build model
    model = create_model(args.net+'_qc', n_classes=args.n_classes)

    if args.resume:
        if os.path.exists(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join(args.resume))

            args.start_epoch = checkpoint['epoch']+1
            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            else:
                raise ValueError("No model or state_dicr Found!!!")

            model.load_state_dict(state_dict)
            print(np.array(checkpoint['regular_mAP']))
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
            
            del checkpoint
            del state_dict
            # del state_dict_ema
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    model.fc.transfer_weight()
    
    if args.is_data_parallel:
        model = torch.nn.DataParallel(model, device_ids=args.parallel_devices)

    model = model.cuda()
    ema_m = ModelEma(model, args.ema_decay)

    # Data loading code
    lb_train_dataset, ub_train_dataset, val_dataset = get_datasets(args)
    print("len(lb_train_dataset):", len(lb_train_dataset)) 
    print("len(ub_train_dataset):", len(ub_train_dataset))
    print("len(val_dataset):", len(val_dataset))

    lb_train_loader = torch.utils.data.DataLoader(
        lb_train_dataset, batch_size=args.lb_bs, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    ub_train_loader = torch.utils.data.DataLoader(
        ub_train_dataset, batch_size=args.ub_bs, shuffle=True,
        num_workers=args.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False,
        num_workers=args.workers, pin_memory=False)


    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, mAPs, mAPs_ema],
        prefix='=> Test Epoch: ')

    # optimizer & scheduler
    optimizer = set_optimizer(model, args)
    args.steps_train = len(ub_train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=args.steps_train, epochs=args.epochs, pct_start=0.2)

    end = time.time()
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    best_mAP = 0
    best_CF1 = 0
    best_OF1 = 0
    thresholds = torch.zeros(args.n_classes).cuda()


    # Used loss
    if args.loss_lb == 'bce':
        criterion_lb = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif args.loss_lb == 'asl':
        criterion_lb = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True, reduction='none')

    
    if args.loss_ub == 'bce':
        criterion_ub = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif args.loss_ub == 'asl':
        criterion_ub = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True, reduction='none')

    # tensorboard
    summary_writer = SummaryWriter(log_dir=args.output)

    ################################################################################################################
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
    args.train_statis = os.path.join(args.output, 'train_statis')
    os.makedirs(args.train_statis, exist_ok=True)
    args.val_statis = os.path.join(args.output, 'val_statis')
    os.makedirs(args.val_statis, exist_ok=True)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
    ################################################################################################################

    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.epochs):

        torch.cuda.empty_cache()

        thresholds = update_ths(lb_train_dataset, ema_m.module, args, logger, thresholds)

        # train for one epoch
        loss = train(lb_train_loader, ub_train_loader, model, ema_m, optimizer, scheduler, epoch, args, logger, criterion_lb, criterion_ub, thresholds)
        
        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # evaluate on validation set
        mAP, CF1, OF1 = validate(val_loader, model, args, logger, epoch, thresholds)
        mAP_ema, CF1_ema, OF1_ema = validate(val_loader, ema_m.module, args, logger, epoch, thresholds)


        mAPs.update(mAP)
        mAPs_ema.update(mAP_ema)
        epoch_time.update(time.time() - end)
        end = time.time()
        eta.update(epoch_time.avg * (args.epochs - epoch - 1))

        regular_mAP_list.append(mAP)
        ema_mAP_list.append(mAP_ema)

        progress.display(epoch, logger)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('val_mAP', mAP, epoch)
            summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)


        # remember best (regular) mAP and corresponding epochs
        if mAP > best_regular_mAP:
            best_regular_mAP = max(best_regular_mAP, mAP)
            best_regular_epoch = epoch
        if mAP_ema > best_ema_mAP:
            best_ema_mAP = max(mAP_ema, best_ema_mAP)
            best_ema_epoch = epoch

        if mAP_ema > mAP:
            mAP = mAP_ema
            CF1 = CF1_ema
            OF1 = OF1_ema

        is_best = mAP > best_mAP
        if is_best:
            best_epoch = epoch
            best_mAP = mAP
            best_CF1 = CF1
            best_OF1 = OF1
            state_dict = model.state_dict()
            state_dict_ema = ema_m.module.state_dict()
            save_checkpoint({
                'epoch': epoch,
                'state_dict': state_dict,
                'state_dict_ema': state_dict_ema,
                'regular_mAP': regular_mAP_list,
                'ema_mAP': ema_mAP_list,
                'best_regular_mAP': best_regular_mAP,
                'best_ema_mAP': best_ema_mAP,
                'optimizer' : optimizer.state_dict(),
            }, is_best=True, filename=os.path.join(args.output, 'best_model.pth.tar'))

        logger.info("{} | Set best mAP {} , CF1 {} , OF1 {} in ep {}".format(epoch, best_mAP, best_CF1, best_OF1, best_epoch))
        logger.info("   | best regular mAP {} in ep {}".format(best_regular_mAP, best_regular_epoch))


        # early stop
        if args.early_stop:
            if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 5:
                if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                    logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                    break

    print("Best mAP:", best_mAP)

    if summary_writer:
        summary_writer.close()
    
    return 0

    
def set_optimizer(model, args):

    if args.optim == 'adam':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    elif args.optim == 'adamw':
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, 'AdamW')(
            param_dicts,
            args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )

    return optimizer

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)

def joint_patch_logits(batch_size, logits_pat, args):
    split_list = torch.split(logits_pat[:], batch_size)                 # [64,80] -> 4 * [16,80]
    logits_joint = torch.stack(split_list, dim=1)                       # 4 * [16,80] -> [16, 4, 80]
    logits_sfmx = torch.softmax(logits_joint, dim=1)                    # [16, {4}, 80]
    logits_joint = (logits_sfmx * logits_joint).sum(dim=1)              # [16, 4, 80] -> [16,80]
    return logits_joint


def train(lb_train_loader, ub_train_loader, model, ema_m, optimizer, scheduler, epoch, args, logger, criterion_lb, criterion_ub, thresholds):
    # record
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    loss_lb_ori = AverageMeter('L_lb_ori', ':5.3f')
    loss_ub_ori = AverageMeter('L_ub_ori', ':5.3f')
    loss_lb_pat = AverageMeter('L_lb_pat', ':5.3f')
    loss_ub_pat = AverageMeter('L_ub_pat', ':5.3f')
    losses = AverageMeter('Loss', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        args.steps_train,
        [loss_lb_ori, loss_ub_ori, loss_lb_pat, loss_ub_pat, lr, losses, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    def get_learning_rate(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    lr.update(get_learning_rate(optimizer))
    logger.info("lr:{}".format(get_learning_rate(optimizer)))

    # switch to train mode
    model.train()
    outputs_ori_list = []
    outputs_pat_list = []
    outputs_mix_list = []
    targets_list = []
    lb_train_iter = iter(lb_train_loader)
    for i, ((inputs_w_ub_list, inputs_s_ub_list), labels_ub) in enumerate(ub_train_loader):

        try:
            (_, inputs_s_lb_list), labels_lb = next(lb_train_iter)
        except:
            lb_train_iter = iter(lb_train_loader)
            (_, inputs_s_lb_list), labels_lb = next(lb_train_iter)

        lb_batch_size = labels_lb.shape[0]
        ub_batch_size = labels_ub.shape[0]

        # cuda
        inputs_w_ub_ori = inputs_w_ub_list[0].cuda(non_blocking=True)
        inputs_w_ub_pat = torch.cat(inputs_w_ub_list[1:], dim=0).cuda(non_blocking=True)
        inputs_s_ub_ori = inputs_s_ub_list[0].cuda(non_blocking=True)
        inputs_s_ub_pat = torch.cat(inputs_s_ub_list[1:], dim=0).cuda(non_blocking=True)
        inputs_s_lb_ori = inputs_s_lb_list[0].cuda(non_blocking=True)
        inputs_s_lb_pat = torch.cat(inputs_s_lb_list[1:], dim=0).cuda(non_blocking=True)
        labels_lb = labels_lb.cuda(non_blocking=True).float()

        # Generate the pseudo labels
        with torch.no_grad():
            outputs_ub_ori = ema_m.module(inputs_w_ub_ori)
            outputs_ub_pat = ema_m.module(inputs_w_ub_pat)

        outputs_ub_ori_ = outputs_ub_ori[0]
        outputs_ub_pat_ = joint_patch_logits(ub_batch_size, outputs_ub_pat[1], args)
        outputs_ub_mix_sm = torch.sigmoid((outputs_ub_ori_ + outputs_ub_pat_) / 2)
        thresholds_flatten = thresholds.unsqueeze(0).repeat(ub_batch_size, 1)
        pseudo_labels_ub = (outputs_ub_mix_sm >= thresholds_flatten).float()

        inputs_ori = torch.cat([inputs_s_lb_ori, inputs_s_ub_ori], dim=0)
        inputs_pat = torch.cat([inputs_s_lb_pat, inputs_s_ub_pat], dim=0)

        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs_ori = model(inputs_ori)
            outputs_pat = model(inputs_pat)    
        
        n_lb_ori = inputs_s_lb_ori.shape[0]
        logits_lb_ori = outputs_ori[0][:n_lb_ori]
        logits_ub_ori = outputs_ori[2][n_lb_ori:]
        Lx_ori = criterion_lb(logits_lb_ori, labels_lb).sum()
        Lu_ori = criterion_ub(logits_ub_ori, pseudo_labels_ub).sum()

        n_lb_pat = inputs_s_lb_pat.shape[0]
        logits_lb_pat = outputs_pat[1][:n_lb_pat]
        logits_ub_pat = outputs_pat[3][n_lb_pat:]

       
        logits_lb_pat_joint = joint_patch_logits(lb_batch_size, logits_lb_pat, args)
        logits_ub_pat_joint = joint_patch_logits(ub_batch_size, logits_ub_pat, args)

        Lx_pat = criterion_lb(logits_lb_pat_joint, labels_lb).sum()
        Lu_pat = criterion_ub(logits_ub_pat_joint, pseudo_labels_ub).sum()
        
        loss = Lx_ori + Lu_ori + Lx_pat + Lu_pat

        # record loss
        loss_lb_ori.update(Lx_ori.item(), lb_batch_size)
        loss_ub_ori.update(Lu_ori.item(), ub_batch_size)
        loss_lb_pat.update(Lx_pat.item(), lb_batch_size)
        loss_ub_pat.update(Lu_pat.item(), ub_batch_size)
        losses.update(loss.item(), lb_batch_size + ub_batch_size)
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # add list
        outputs_ori_list.append(torch.sigmoid(outputs_ub_ori_).cpu().numpy())
        outputs_pat_list.append(torch.sigmoid(outputs_ub_pat_).cpu().numpy())
        outputs_mix_list.append(outputs_ub_mix_sm.cpu().numpy())
        targets_list.append(labels_ub.numpy())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # one cycle learning rate
        scheduler.step()
        lr.update(get_learning_rate(optimizer))
        ema_m.update(model)

        if i % args.print_freq == 0:
            progress.display(i, logger)

    ################################################################################################################
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
    outputs_ori_np = np.concatenate(outputs_ori_list)
    outputs_pat_np = np.concatenate(outputs_pat_list)
    outputs_mix_np = np.concatenate(outputs_mix_list)
    targets_np = np.concatenate(targets_list)
    thresholds_np = thresholds.cpu().numpy()
    np.save(os.path.join(args.train_statis, f"epoch_{epoch}_outputs_ori"), outputs_ori_np)
    np.save(os.path.join(args.train_statis, f"epoch_{epoch}_outputs_pat"), outputs_pat_np)
    np.save(os.path.join(args.train_statis, f"epoch_{epoch}_outputs_mix"), outputs_mix_np)
    np.save(os.path.join(args.train_statis, f"epoch_{epoch}_targets"), targets_np)
    np.save(os.path.join(args.train_statis, f"epoch_{epoch}_thresholds"), thresholds_np)
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
    ################################################################################################################

    return losses.avg



@torch.no_grad()
def validate(val_loader, model, args, logger, epoch, thresholds):
    batch_time = AverageMeter('Time', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, mem],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    outputs_ori_list = []
    outputs_pat_list = []
    outputs_mix_list = []
    targets_list = []
        
    end = time.time()
    for i, (inputs_list, targets) in enumerate(val_loader):
        
        batch_size = targets.shape[0]

        inputs_ori = inputs_list[0].cuda(non_blocking=True)
        inputs_pat = torch.cat(inputs_list[1:], dim=0).cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs_ori = model(inputs_ori)
            outputs_pat = model(inputs_pat)

        
        outputs_ori_ = outputs_ori[0]
        outputs_pat_ = joint_patch_logits(batch_size, outputs_pat[1], args)
        outputs_mix_ = (outputs_ori_ + outputs_pat_) / 2

        outputs_ori_sm = torch.sigmoid(outputs_ori_)
        outputs_pat_sm = torch.sigmoid(outputs_pat_)
        outputs_mix_sm = torch.sigmoid(outputs_mix_)

        # add list
        outputs_ori_list.append(outputs_ori_sm.detach().cpu())
        outputs_pat_list.append(outputs_pat_sm.detach().cpu())
        outputs_mix_list.append(outputs_mix_sm.detach().cpu())
        targets_list.append(targets.detach().cpu())

        # record memory
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    labels = np.concatenate(targets_list)
    outputs_ori = np.concatenate(outputs_ori_list)
    outputs_pat = np.concatenate(outputs_pat_list)
    outputs_mix = np.concatenate(outputs_mix_list)

    # calculate mAP
    mAP_ori = function_mAP(labels, outputs_ori)
    mAP_pat = function_mAP(labels, outputs_pat)
    mAP_mix = function_mAP(labels, outputs_mix)
    
    print("Calculating mAP:")  
    logger.info("  mAP_ori: {}".format(mAP_ori))
    logger.info("  mAP_pat: {}".format(mAP_pat))
    logger.info("  mAP_mix: {}".format(mAP_mix))

    ################################################################################################################
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
    thresholds_np = thresholds.cpu().numpy()
    np.save(os.path.join(args.val_statis, f"epoch_{epoch}_outputs_ori"), outputs_ori)
    np.save(os.path.join(args.val_statis, f"epoch_{epoch}_outputs_pat"), outputs_pat)
    np.save(os.path.join(args.val_statis, f"epoch_{epoch}_outputs_mix"), outputs_mix)
    np.save(os.path.join(args.val_statis, f"epoch_{epoch}_targets"), labels)
    np.save(os.path.join(args.val_statis, f"epoch_{epoch}_thresholds"), thresholds_np)
    #|||||||||||||||||||||||||||||||||||||||||||||||Test: OF1 CF1||||||||||||||||||||||||||||||||||||||||||||||||||#
    thresholds_flatten = np.expand_dims(thresholds_np, 0).repeat(outputs_mix.shape[0], axis=0)
    pseudo_labels = (outputs_mix >= thresholds_flatten).astype(np.float32)
    
    n_correct_pos = (labels*pseudo_labels).sum(0)
    n_pred_pos = ((pseudo_labels==1)).sum(0)
    n_true_pos = labels.sum(0)
    OP = n_correct_pos.sum()/n_pred_pos.sum()
    CP = np.nanmean(n_correct_pos/n_pred_pos)
    OR = n_correct_pos.sum()/n_true_pos.sum()
    CR = np.nanmean(n_correct_pos/n_true_pos)

    CF1 = (2 * CP * CR) / (CP + CR)
    OF1 = (2 * OP * OR) / (OP + OR)

    logger.info("  CF1: {}".format(CF1))
    logger.info("  OF1: {}".format(OF1))
    #||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||#
    ################################################################################################################

    return mAP_mix, CF1 * 100, OF1 * 100


@torch.no_grad()
def update_ths(lb_train_dataset, model, args, logger, thresholds):
    loader = torch.utils.data.DataLoader(
        lb_train_dataset, batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    batch_time = AverageMeter('Time', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(loader),
        [batch_time, mem],
        prefix='Update thresholds: ')

    # switch to evaluate mode
    model.eval()
    outputs_list = []
    targets_list = []
        
    end = time.time()
    for i, ((inputs_w_list, _), targets) in enumerate(loader):

        batch_size = targets.shape[0]

        inputs_ori = inputs_w_list[0].cuda(non_blocking=True)
        inputs_pat = torch.cat(inputs_w_list[1:], dim=0).cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        # mixed precision ---- compute outputs
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs_ori = model(inputs_ori)
            outputs_pat = model(inputs_pat)

        outputs_ori_ = outputs_ori[0]
        outputs_pat_ = joint_patch_logits(batch_size, outputs_pat[1], args)
        outputs_mix_sm = torch.sigmoid((outputs_ori_ + outputs_pat_) / 2)

        # add list
        outputs_list.append(outputs_mix_sm.detach().cpu())
        targets_list.append(targets.detach().cpu())

        # record memory
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)
    
    targets_tensor = torch.cat(targets_list).cuda()
    outputs_tensor = torch.cat(outputs_list).cuda()

    best_me = torch.zeros(targets_tensor.shape[1]).cuda()
    best_th = torch.zeros(targets_tensor.shape[1]).cuda()
    if args.cat_strategy == 'f1':
        for th in np.arange(0.00, 1.00, 0.01):
            pseudo_labels = (outputs_tensor >= th).float()
            n_correct_pos = (targets_tensor*pseudo_labels).sum(0)
            n_pred_pos = ((pseudo_labels==1)).sum(0)
            n_true_pos = targets_tensor.sum(0)
            P = n_correct_pos/n_pred_pos
            R = n_correct_pos/n_true_pos
            F1 = ((2 * P * R) / (P + R)).float()
            update_idx = (F1 > best_me)
            best_me[update_idx] = F1[update_idx]
            best_th[update_idx] = th
    elif args.cat_strategy == 'precision':
        for th in np.arange(0.00, 1.00, 0.01):
            pseudo_labels = (outputs_tensor >= th).float()
            n_correct_pos = (targets_tensor*pseudo_labels).sum(0)
            n_pred_pos = ((pseudo_labels==1)).sum(0)
            P = (n_correct_pos/n_pred_pos).float()
            update_idx = (P > best_me)
            best_me[update_idx] = P[update_idx]
            best_th[update_idx] = th
    elif args.cat_strategy == 'recall':
        for th in np.arange(0.00, 1.00, 0.01):
            pseudo_labels = (outputs_tensor >= th).float()
            n_correct_pos = (targets_tensor*pseudo_labels).sum(0)
            n_true_pos = targets_tensor.sum(0)
            R = (n_correct_pos/n_true_pos).float()
            update_idx = (R > best_me)
            best_me[update_idx] = R[update_idx]
            best_th[update_idx] = th
    elif args.cat_strategy == 'fB':
        for th in np.arange(0.00, 1.00, 0.01):
            pseudo_labels = (outputs_tensor >= th).float()
            n_correct_pos = (targets_tensor*pseudo_labels).sum(0)
            n_pred_pos = ((pseudo_labels==1)).sum(0)
            n_true_pos = targets_tensor.sum(0)
            P = n_correct_pos/n_pred_pos
            R = n_correct_pos/n_true_pos
            FB = (((1 + args.beta * args.beta) * P * R) / (args.beta * P + R)).float()
            update_idx = (FB > best_me)
            best_me[update_idx] = FB[update_idx]
            best_th[update_idx] = th
    thresholds = best_th

    logger.info("  Thresholds: {}".format(thresholds))

    return thresholds


if __name__ == '__main__':
    main()