import timm
import torch
from PIL import Image
import numpy as np
from torch import tensor
import torch.nn.functional as F
import os
from tqdm import tqdm
from typing import Any, Dict, List, Tuple
from torchvision import transforms 
from glob import glob 

Image.MAX_IMAGE_PIXELS = None

img_feats_target, mask_feats_target = "./datasets/DigestPath/patch_feats/img", "./datasets/DigestPath/patch_feats/mask"
prototype_feats_target = './datasets/DigestPath/patch_feats/prototype'
img_pic_target, mask_pic_target = "./datasets/DigestPath/patches/img", "./datasets/DigestPath/patches/mask"
prototype_pic_target = './datasets/DigestPath/patches/prototype'
img_root, mask_root = "datasets/DigestPath/raw_data/img", "datasets/DigestPath/raw_data/mask"
trans = transforms.Compose( [
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225))
    ]
)
def preprocess(img, type=False):
    if type:
        h, w = img.shape[:-1]
        h_p, w_p = ((h - 1) // 512 + 1) * 512, ((w - 1) // 512 + 1) * 512
        img= np.pad(img, ((0, h_p-h), (0, w_p-w),(0,0)))
    else: 
        h, w = img.shape
        h_p, w_p = ((h - 1) // 512 + 1) * 512, ((w - 1) // 512 + 1) * 512
        img= np.pad(img, ((0, h_p-h), (0, w_p-w)))
    return img

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
encoder = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
uni_dict = torch.load('./pytorch_model.bin',map_location="cpu")
encoder.load_state_dict(uni_dict)
encoder = encoder.to(device)
encoder.eval()
# os.makedirs(img_target, exist_ok=True)
# os.makedirs(mask_target,exist_ok=True)

# for index in tqdm(os.listdir(img_root)):
    # os.makedirs(os.path.join(img_feats_target,index[:-4]), exist_ok=True)
    # os.makedirs(os.path.join(img_pic_target,index[:-4]),exist_ok=True)

    # img = Image.open(os.path.join(img_root, index))
    # img = np.array(img)
    # img = preprocess(img,True)
    # h, w = img.shape[:-1]
#     for i in range(0, h, 512):
#         for j in range(0, w, 512):
#             patch = img[i:i+512, j:j+512,:]
#             patch = Image.fromarray(patch,'RGB')
#             patch.save(os.path.join(img_pic_target, index[:-4], str(i) + "_" + str(j) + ".png"))
#             torch.cuda.empty_cache()
    
    # for i in range(0, h, 512):
    #     for j in range(0, w, 512):
    #         patch = img[i:i+512, j:j+512,:]
    #         patch_tensor = trans(patch)
    #         patch_tensor = patch_tensor.to(device)
    #         patch_tensor = patch_tensor.unsqueeze(0)
    
    #         feature = encoder.forward_features(patch_tensor).detach().cpu()
    #         torch.save(feature[0,0,:],os.path.join(img_feats_target,index[:-4],str(i)+"_"+str(j)+"_cls.pt"))
    #         torch.save(feature[0,1:,:],os.path.join(img_feats_target,index[:-4],str(i)+"_"+str(j)+"_local.pt"))
    #         torch.cuda.empty_cache()
for mask_dir in tqdm(glob(os.path.join(mask_pic_target,'*/*.png'))):
    idx = mask_dir.split(os.sep)[-2]
    # os.makedirs(os.path.join(prototype_pic_target,idx),exist_ok=True)
    # os.makedirs(os.path.join(prototype_feats_target,idx),exist_ok=True)
    patch_idx = mask_dir.split(os.sep)[-1].split('.')[0]
    i,j = patch_idx.split('_')[0], patch_idx.split('_')[1]
    mask = Image.open(mask_dir).convert('RGB')
    img_dir = mask_dir.replace('mask','img')
    img = Image.open(img_dir)
    prototype_array = (np.array(mask) / 255).astype(np.uint8) * np.array(img)
    print(np.array(img).shape)
    print(np.array(mask).shape)
    print(prototype_array.shape)
    prototype_img = Image.fromarray(prototype_array, 'RGB')
    print(prototype_img.size)
    # prototype_img.save(mask_dir.replace('mask','prototype')) 
    # masked_img_tensor = trans(prototype_img).unsqueeze(0).to(device)
    # feature = encoder.forward_features(masked_img_tensor).detach().cpu()
    # torch.save(feature[0,0,:],os.path.join(prototype_feats_target,idx,str(i)+"_"+str(j)+"_cls.pt"))
    break
    
for index in tqdm(os.listdir(mask_root)):
    os.makedirs(os.path.join(mask_feats_target,index[:-4]), exist_ok=True)
    os.makedirs(os.path.join(prototype_feats_target,index[:-4]), exist_ok=True)
    os.makedirs(os.path.join(prototype_pic_target,index[:-4]), exist_ok=True)
    os.makedirs(os.path.join(mask_pic_target,index[:-4]),exist_ok=True)
    img = Image.open(os.path.join(img_root, index.replace('.png','.jpg'))).convert('RGB')
    img = np.array(img)
    img = preprocess(img,True)
    mask = Image.open(os.path.join(mask_root, index)).convert('L')
    mask = np.array(mask) / 255
    mask= preprocess(mask)
    h, w = mask.shape
    print(index)
    # print(img_tensor.shape)
    for i in range(0, h, 512):
        for j in range(0, w, 512):
            patch = mask[i:i+512, j:j+512]

    for i in range(0, h, 512):
        for j in range(0, w, 512):
            patch_mask = mask[i:i+512, j:j+512]
            patch_mask = np.expand_dims(patch_mask, axis=2)
            patch_img = img[i:i+512, j:j+512,:]
            patch_masked_img = patch_img * patch_mask
            prototype_img =  Image.fromarray(patch_masked_img,'RGB')
            prototype_img.save(os.path.join(prototype_pic_target, index[:-4], str(i) + "_" + str(j) + ".png"))
            masked_img_tensor = trans(patch_masked_img).unsqueeze(0).to(device)
            masked_img_tensor = torch.repeat_interleave(masked_img_tensor,3,dim=1).to(device)
            feature = encoder.forward_features(masked_img_tensor).detach().cpu()
            torch.save(feature[0,0,:],os.path.join(prototype_feats_target,index[:-4],str(i)+"_"+str(j)+"_cls.pt"))
            torch.save(feature[0,1:,:],os.path.join(mask_feats_target,index[:-4],str(i)+"_"+str(j)+"_local.pt"))
            torch.cuda.empty_cache()
    break