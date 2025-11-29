import os
import subprocess
import sys
import torch
import argparse
from functools import partial
import torch.distributed
from torchvision.ops import nms
import yaml
import logging
import pprint
from transformers import AutoImageProcessor, AutoModel
# from datasets_pro import DigestDataset
# from dataset_semiseg import DigestDataset_label, DigestDataset_unlabel
from eval import DiceLoss, Jaccard, UncertainMinimization
import torch.nn as nn
import numpy as np
import random
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
from SemiSeg.models.ssr_sam_decoder import SSR_Decoder
from SemiSeg.models.ssr_sam_model import SSR_model

# from torchvision import transforms
import time
import timm
import json
from copy import deepcopy
from glob import glob
from monai.metrics import DiceMetric, compute_iou
from models.encoder import ImageEncoderViT
from SemiSeg.dataset_semiseg import Dataset_semiseg
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed
import timm
from torch.optim import AdamW

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_activate_semi(model):
    for name, param in model.named_parameters():
        # if name.startswith("module.encoder") and 'lora' not in name:
        if name.startswith("module.encoder"):
            param.requires_grad = False
        else:
            param.requires_grad = True
        # if any(f'blocks.{i}' in name for i in [8,9,10,11]) and 'lora' in name:
        #     param.requires_grad = True
    return model


def create_model(cfg,args):
    
    print("initialing model")
    isemi_model = SSR_model(
        # encoder = AutoModel.from_pretrained('ckpt/DINOv2/dino-base').cuda(),
        encoder = ImageEncoderViT(
            depth=12,
            embed_dim=768,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=12,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[2, 5, 8, 11],
            window_size=14,
            ),
        decoder=SSR_Decoder(
            num_classes=cfg['nclass'],
            image_embed_dim=cfg['encoder_dim'],
            image_embed_hw=(64, 64),
            transformer_dim=256,
            global_dim=cfg["global_dim"]
            
        ),
        original_imgsize=(cfg['crop_size'], cfg['crop_size'])
    )
    # model_dict = isemi_model.state_dict()
    # from peft import LoraConfig, TaskType, get_peft_model
    # modules = [
    #     f"encoder.blocks.{i}.{layer}"
    #     for i in range(12)
    #     for layer in ["attn.qkv", "attn.proj", "mlp.lin1", "mlp.lin2"]
    # ]
    # config = LoraConfig(target_modules=modules)
    # get_peft_model(isemi_model, config)
    if cfg["pretrain"]:
        print("loading pretrain model")
        checkpoint = torch.load(args.pretrain_path,map_location="cpu")
        state_dict = checkpoint['model']
        new_state_dict = {}
        for k,v in state_dict.items():
            new_state_dict[k[7:]] = v
        isemi_model.load_state_dict(new_state_dict)
        previous_best = checkpoint['previous_best']
        print('previous_best: ', previous_best)
    
    return isemi_model

def sample_label_prototype(label_prototype, sample_n=4):
    sampled_prototypes = []
    # non_mask_prototype = torch.load('./none_mask_prototype.pt')
    for i in range(label_prototype.shape[0]):
        cls_prototype = torch.unique(label_prototype[i,:,:],dim=0).detach()
        # cls_prototype = cls_prototype[(cls_prototype != non_mask_prototype).any(dim=1)]
        cls_prototype = cls_prototype.index_select(dim=0, index = torch.tensor(random.choices(range(cls_prototype.shape[0]),k=sample_n)))
        sampled_prototypes.append(cls_prototype)
    s = torch.stack(sampled_prototypes,dim=0)
    return s

def evaluate(val_loader, model, label_prototype, cfg, args):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    world_size = dist.get_world_size()
    for img, mask, id in val_loader:
        img, mask = img.cuda(), mask
        batched_label_prototype = label_prototype.expand(
            img.shape[0], -1, -1, -1
        ).cuda()
        with torch.no_grad():
            preds_logits = model(img, batched_label_prototype)
        preds = torch.argmax(preds_logits, dim=1)
        # if args.local_rank == 0:
        #     print("preds_logits",preds_logits[0,:,0,0])
        #     print('preds',torch.max(preds))
        #     print('mask',torch.max(mask))

        intersection, union, target = \
                intersectionAndUnion(preds.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
        
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)

        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)
    return mIOU, iou_class


def setup_distributed(backend="nccl", port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "10685"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return rank, world_size


def main(args):
    cfg = yaml.load(open(args.config,"r"), Loader=yaml.Loader)  
    rank, world_size = setup_distributed(port=args.port)
    if rank == 0:
        logger = init_log('global', logging.INFO)
        logger.propagate = 0
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        os.makedirs(args.log_dir,exist_ok=True)

    torch.cuda.set_device(args.local_rank)
    cudnn.enabled = True
    cudnn.benchmark = True
    device = torch.device('cuda:{}'.format(args.local_rank))
    isemi_model = create_model(cfg,args)
    optimizer = torch.optim.AdamW(isemi_model.parameters(), lr=cfg["lr"])
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(isemi_model)))
    isemi_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(isemi_model)
    isemi_model = isemi_model.to(device)
    isemi_model = DDP(isemi_model, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs'])
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    
    criterion_u = nn.CrossEntropyLoss(reduction='none')
    trainset_u = Dataset_semiseg(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = Dataset_semiseg(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    val = Dataset_semiseg(cfg['dataset'], cfg['data_root'], 'val',val_path=args.val_path)

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(val)
    valloader = DataLoader(val, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)
    # best_model = deepcopy(isemi_model) 
    # best_model = best_model.to(device)

    # label_prototype  nc*n*d
    label_prototype = torch.load(args.global_prototype_path,map_location='cpu', weights_only=True)
    label_prototype = sample_label_prototype(label_prototype,cfg['mask_prompt_num'])
    label_prototype = label_prototype.unsqueeze(0).cuda()

    total_iters = len(trainloader_l) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.log_dir, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.log_dir, 'latest.pth'))
        isemi_model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    # miou, iou_class = evaluate(valloader, isemi_model, label_prototype, cfg, args)
    # if rank == 0:
    #         for cls_idx, iou in enumerate(iou_class):
    #             logger.info(
    #                 "***** Evaluation ***** >>>> Class [{:} {:}] "
    #                 "IoU: {:.2f}".format(
    #                     cls_idx, CLASSES[cfg["dataset"]][cls_idx], iou
    #                 )
    #             )
    #         logger.info(
    #             "***** Evaluation epoch{} ***** >>>> MeanIoU: {:.2f}\n".format(
    #                 -1, miou
    #             )
    #         )

    for epoch in range(epoch+1,5):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))
        
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_loss_pp = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        # train_loss = train_supervised(args, epoch, train_l_loader, isemi_model, optimizer, CE_criterion, dice_metric, label_prototype)
        isemi_model = train_activate_semi(isemi_model)

        for i, ((img, mask),
            (img_w, img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2, indexs),
            (img_w_mix, img_s1_mix, img_s2_mix, ignore_mask_mix, _, _, _)) in enumerate(loader):

            batched_label_prototype = label_prototype.expand(img.shape[0],-1,-1,-1).cuda()
            img, mask = img.cuda(), mask.cuda()
            img_w, img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2 = img_w.cuda(), img_s1.cuda(), img_s2.cuda(), ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            img_w_mix, img_s1_mix, img_s2_mix, ignore_mask_mix = img_w_mix.cuda(), img_s1_mix.cuda(), img_s2_mix.cuda(), ignore_mask_mix.cuda()
            
            # 将一个batch的P_wsi pad成相同维度
            # if args.local_rank == 0 : print(len(indexs))
            # prototype_wsi_list = [prototype_wsi_dict[index] for index in indexs]
            # max_len = max([i.shape[1] for i in prototype_wsi_list])
            # padded_list =  [torch.cat((i, torch.zeros(1, max_len - i.shape[1], args.feats_dim, device=device)),dim=1) for i in prototype_wsi_list]
            # padded_list =  [torch.cat((i, zero_cls.expand(-1,max_len - i.shape[1],-1)),dim=1) for i in prototype_wsi_list]
            # prototype_wsi = torch.stack(prototype_wsi_list, dim=0)

            with torch.no_grad():
                isemi_model.eval()
                pred_w_mix = isemi_model(img_w_mix, batched_label_prototype).detach()

            isemi_model.train()
            
            # 将img_s1,img_s2 cutmix
            cutmix_box1_3c = cutmix_box1.unsqueeze(1).expand(img_s1.shape)   # 1 channel  to 3 channels
            cutmix_box2_3c = cutmix_box2.unsqueeze(1).expand(img_s2.shape)

            img_s1[cutmix_box1.unsqueeze(1).expand(img_s1.shape)==1] = img_s1_mix[cutmix_box1.unsqueeze(1).expand(img_s1.shape)==1]
            img_s2[cutmix_box2.unsqueeze(1).expand(img_s2.shape)==1] = img_s2_mix[cutmix_box2.unsqueeze(1).expand(img_s2.shape)==1]

            preds = isemi_model(img, batched_label_prototype)

            preds_w = isemi_model(img_w, batched_label_prototype).detach()
            conf_w = preds_w.softmax(dim=1).max(dim=1)[0]
            masks_w = preds_w.argmax(dim=1)
            
            preds_fp = isemi_model(img_w, batched_label_prototype, need_fp=True)

            preds_s1 = isemi_model(img_s1, batched_label_prototype)

            preds_s2 = isemi_model(img_s2, batched_label_prototype)

            dim_size = batched_label_prototype.size(2)
            random_indices = torch.randint(0, dim_size, (int(np.ceil(dim_size//8)),)).cuda()
            chosen_prototype = batched_label_prototype.index_select(2, random_indices)
            preds_pp = isemi_model(img_w, chosen_prototype)
            # if args.local_rank == 0: print_gpu_memory()

            preds_w_cutmixed1, preds_w_cutmixed2 = preds_w.clone(), preds_w.clone()
            ignore_mask_cutmixed1, ignore_mask_cutmixed2 = ignore_mask.clone(), ignore_mask.clone()
            # 将pred_w和pred_w_mix cutmix
            preds_w_cutmixed1[cutmix_box1.unsqueeze(1).expand(-1,cfg["nclass"],-1,-1) == 1] = pred_w_mix[cutmix_box1.unsqueeze(1).expand(-1,cfg["nclass"],-1,-1) == 1]
            preds_w_cutmixed2[cutmix_box2.unsqueeze(1).expand(-1,cfg["nclass"],-1,-1) == 1] = pred_w_mix[cutmix_box2.unsqueeze(1).expand(-1,cfg["nclass"],-1,-1) == 1]
            conf_w_cutmixed1, masks_w_cutmixed1 = preds_w_cutmixed1.softmax(dim=1).max(dim=1)[0], preds_w_cutmixed1.argmax(dim=1)
            conf_w_cutmixed2, masks_w_cutmixed2 = preds_w_cutmixed2.softmax(dim=1).max(dim=1)[0], preds_w_cutmixed2.argmax(dim=1)
    

            ignore_mask_cutmixed1[cutmix_box1.squeeze(1) == 1] = ignore_mask_mix[cutmix_box1.squeeze(1) == 1]
            ignore_mask_cutmixed2[cutmix_box2.squeeze(1) == 1] = ignore_mask_mix[cutmix_box2.squeeze(1) == 1]

            # mask = torch.argmax(mask,dim=1)
            loss_l = criterion_l(preds, mask).mean()

            loss_u_s1 = criterion_u(preds_s1, masks_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_w_cutmixed1 >= cfg['confidence_threshold']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum()/(ignore_mask_cutmixed1 != 255).sum()

            loss_u_s2 = criterion_u(preds_s2, masks_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_w_cutmixed2 >= cfg['confidence_threshold']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum()/(ignore_mask_cutmixed2 != 255).sum()

            loss_fp = criterion_u(preds_fp, masks_w)
            loss_fp = loss_fp * ((conf_w >= cfg['confidence_threshold']) & (ignore_mask != 255))
            loss_fp = loss_fp.sum()/ (ignore_mask != 255).sum()

            loss_pp = criterion_u(preds_pp, masks_w)
            loss_pp = loss_pp * ((conf_w >= cfg['confidence_threshold']) & (ignore_mask != 255))
            loss_pp = loss_pp.sum()/ (ignore_mask != 255).sum()

            # loss = (loss_l + loss_fp * 0.5 + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 )
            # loss = loss_l + loss_pp
            loss = (loss_l + loss_fp * 0.5 + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_pp)
            torch.distributed.barrier()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_l.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_fp.item())
            total_loss_pp.update(loss_pp.item())

            mask_ratio = ((conf_w >= cfg['confidence_threshold']) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_l) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            # optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            if i % 1000 == 0 and rank == 0:
                logger.info('Iters: {:}/{:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Loss_pp:{:.3f} Mask ratio: '
                            '{:.3f}'.format(i, len(trainloader_l), total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                                            total_loss_w_fp.avg, total_loss_pp.avg, total_mask_ratio.avg))
            torch.cuda.empty_cache()
        if epoch % 1 == 0:
            miou, iou_class = evaluate(valloader,isemi_model,label_prototype,cfg,args)
            if rank == 0:
                for (cls_idx, iou) in enumerate(iou_class):
                    logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                                'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
                logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format('original', miou))

            is_best = miou > previous_best
            previous_best = max(miou, previous_best)
            if rank == 0:
                checkpoint = {
                    'model': isemi_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'previous_best': previous_best,
                }
                torch.save(checkpoint, os.path.join(args.log_dir, 'latest.pth'))
                if is_best:
                    torch.save(checkpoint, os.path.join(args.log_dir, 'best.pth'))

        torch.distributed.barrier()
        torch.cuda.empty_cache()

def parse():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--config", default="./configs/bcss_supervised.yaml", type=str, help="")
    parser.add_argument('--pretrain_path', type=str, required=True)
    parser.add_argument('--global_prototype_path', type=str, required=True)
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--val-path', type=str, required=True)
    parser.add_argument("--log_dir", default="./exp", type=str, help="")
    parser.add_argument("--random_seed", default=8, type=int, help="")
    parser.add_argument("--port", default=None, type=int)
    parser.add_argument(
        "--local-rank", default=-1, type=int, help="node rank for distributed training"
    )

    args = parser.parse_args()
    return args


if __name__ =='__main__':
    args = parse()
    setup_seed(args.random_seed)
    main(args)
