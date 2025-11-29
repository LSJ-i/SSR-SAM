import random
import sys 
import argparse
from monai.metrics import DiceMetric,compute_iou
from PIL import Image
import torch 
import os 
import numpy as np 
from tqdm import tqdm
import yaml
from glob import glob 
from torchvision import transforms
from SemiSeg.models.ssr_sam_decoder import SSR_Decoder
from SemiSeg.models.ssr_sam_model import SSR_model
from models.encoder import ImageEncoderViT
from functools import partial
from SemiSeg.dataset_semiseg import Dataset_semiseg
import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP


# from model.semseg.dpt import DPT



def main(args):
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)
    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
    isemi_model = SSR_model(
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
    

    from peft import LoraConfig, TaskType, get_peft_model

    modules = [
        f"encoder.blocks.{i}.{layer}"
        for i in range(12)
        for layer in ["attn.qkv", "attn.proj", "mlp.lin1", "mlp.lin2"]
    ]
    config = LoraConfig(target_modules=modules)
    get_peft_model(isemi_model, config)
    checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'),map_location="cpu")

    # checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
    state_dict = checkpoint['model']
    
    new_state_dict = {}
    for k,v in state_dict.items():
        new_state_dict[k[7:]] = v
    isemi_model.load_state_dict(new_state_dict)
    previous_best = checkpoint['previous_best']
    print('previous_best: ',previous_best)
    device = torch.device("cuda:{}".format(args.local_rank))
    isemi_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(isemi_model)
    isemi_model = isemi_model.to(device)
    isemi_model = DDP(
        isemi_model,
        device_ids=[args.local_rank],
        output_device=args.local_rank,
        find_unused_parameters=True,
    )
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    val = Dataset_semiseg(cfg["dataset"], cfg["data_root"], "val", val_path=args.test_path)
    valsampler = torch.utils.data.distributed.DistributedSampler(val)
    valloader = DataLoader(val, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)
    isemi_model.eval()
    with torch.no_grad():
        for img, mask, id in tqdm(valloader):
            img = img.cuda()
            label_prototype = torch.load(
            args.global_path, map_location="cpu", weights_only=True)
            label_prototype = label_prototype.unsqueeze(0).to(device)
            batched_label_prototype = label_prototype.expand(
            img.shape[0], -1, -1, -1
            ).cuda()
            pred = isemi_model(img,batched_label_prototype).argmax(dim=1)
    
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
        mIoU = np.mean(iou_class)
    
    if rank == 0:
        for (cls_idx, iou) in enumerate(iou_class):
            logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                        'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
        logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format("", mIoU))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semi-Supervised Semantic Segmentation")
    parser.add_argument("--config", type=str, default="configs/bcss_supervised.yaml")
    parser.add_argument('--save-path', type=str, default="exp/bcss/train_s3sam/853")
    # parser.add_argument('--save-path', type=str, default="exp/inria/_with_lora/sam/0.08")
    parser.add_argument('--test_path', type=str, default="/remote-home/share/cym/SemiSeg/datasets/BCSS/split/test.txt")
    parser.add_argument('--global_path', type=str, default="datasets/BCSS/split/853/global.pt")
    parser.add_argument('--local-rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    args = parser.parse_args()
    main(args)
    