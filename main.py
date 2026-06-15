import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from SWMF import denorm_to_01, SideWindowMeanFilter
from matplotlib import cm
import time
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import InterpolationMode
import torchvision.transforms as T
import torchvision.transforms.functional as TF


import torch.nn.functional as F


import pandas as pd
from PIL import Image
import sys

import torch
import argparse
import torch.optim as optim
import logging
import datetime
import random
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from torch import nn
from tqdm import tqdm





def worker_init_fn(worker_id):
    import cv2
    cv2.setNumThreads(0)
    torch.set_num_threads(1)

def str2bool(v):
    if isinstance(v, bool): return v
    v = v.lower()
    if v in ("yes","true","t","1","y"): return True
    if v in ("no","false","f","0","n"): return False
    raise argparse.ArgumentTypeError("Boolean value expected.")
def config():
    parser = argparse.ArgumentParser(description='classification implementation')
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dataset_name", type=str, default="ISIC2018")
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument('--save_folder', default='checkpoints',
                        help='Location to save checkpoint models')
    parser.add_argument('--show_interval', default=50, type=int, help='interval of showing training conditions')
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument('--pseudo_mode', type=str, default="hard")
    parser.add_argument('--use_spectral', type=str2bool, default=True)
    parser.add_argument('--use_gnn', type=str2bool, default=True)
    parser.add_argument('--use_filter', type=str2bool, default=True)
    parser.add_argument('--pseudo_thr', type=float, default="0.5")
    parser.add_argument("--save_seg224", type=bool, default=False)
    parser.add_argument("--seg_save_root", type=str, default="./seg_outputs")
    parser.add_argument("--save_cam_vis", type=bool, default=False)
    parser.add_argument("--cam_cmap", type=str, default="jet")
    parser.add_argument("--cam_alpha", type=float, default=0.45)
    parser.add_argument("--eval_ckpt", type=str, default="", help="Path to checkpoint for one-time evaluation.")
    parser.add_argument("--eval_only", action="store_true", help="Only run validation once and exit.")
    parser.add_argument("--eval_strict", type=str2bool, default=False, help="Whether to load checkpoint with strict=True.")


    return parser.parse_args()


def get_model(args):
    if args.dataset_name == "PH2":
        num_classes = 3
        lr = 1e-5
        eta_min = 1e-7
        beta_scale = 0.1
    elif args.dataset_name == "ISIC2017":
        num_classes = 3
        lr = 1e-4
        eta_min = 1e-6
        beta_scale = 0.99
    elif args.dataset_name == "ISIC2018":
        num_classes = 4
        lr = 1e-4
        eta_min = 1e-6
        beta_scale = 1.0
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    model = IntegratedSkinNet(
        num_classes=num_classes,
        num_clusters=8,
        use_spectral=args.use_spectral,
        use_gnn=args.use_gnn,
        use_filter=args.use_filter,
        beta_scale=beta_scale,
    )
    print(f"[CONFIG] dataset={args.dataset_name}, beta_scale={beta_scale}")

    
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=eta_min)
    scaler = GradScaler()

    return model, optimizer, scheduler, scaler


def _smart_load_state_dict(model, state_dict, strict=False):
    model_sd = model.state_dict()
    model_keys = list(model_sd.keys())
    ckpt_keys = list(state_dict.keys())

    ckpt_has_module = len(ckpt_keys) > 0 and ckpt_keys[0].startswith("module.")
    model_has_module = len(model_keys) > 0 and model_keys[0].startswith("module.")

    if ckpt_has_module and not model_has_module:
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    elif (not ckpt_has_module) and model_has_module:
        state_dict = {("module." + k): v for k, v in state_dict.items()}

    incompatible = model.load_state_dict(state_dict, strict=strict)

    if hasattr(incompatible, "missing_keys") and hasattr(incompatible, "unexpected_keys"):
        print(f"[CKPT] missing keys: {len(incompatible.missing_keys)}")
        print(f"[CKPT] unexpected keys: {len(incompatible.unexpected_keys)}")
        if len(incompatible.missing_keys) > 0:
            print(f"[CKPT] missing first 20: {incompatible.missing_keys[:20]}")
        if len(incompatible.unexpected_keys) > 0:
            print(f"[CKPT] unexpected first 20: {incompatible.unexpected_keys[:20]}")


def load_checkpoint_to_model(model, ckpt_path, device="cuda", strict=False):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state_dict = ckpt

    _smart_load_state_dict(model, state_dict, strict=strict)
    model.to(device)
    model.eval()

    print(f"[EVAL_ONCE] Loaded checkpoint: {ckpt_path}")
    return model


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PH2Dataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir=None, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

        self.pre_ops = [
            ("orig",  False,   0),
            ("rot90", False, -90),
            ("rot180",False, -180),
            ("rot270",False, -270),
            ("flip",  True,    0),
            ("flip_rot90",  True,  -90),
            ("flip_rot180", True, -180),
            ("flip_rot270", True, -270),
        ]

    def __len__(self):
        return len(self.annotations) * len(self.pre_ops)

    def __getitem__(self, idx):
        num_ops = len(self.pre_ops)
        base_idx = idx // num_ops
        op_idx = idx % num_ops
        op_name, do_flip, rot_angle = self.pre_ops[op_idx]

        image_id = self.annotations.iloc[base_idx, 0]
        dx = self.annotations.iloc[base_idx, 1]
        label = self.get_label(dx)

        img_path = os.path.join(self.img_dir, f"{image_id}.bmp")
        image = Image.open(img_path).convert("RGB")

        mask = None
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, f"{image_id}.bmp")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
            else:
                mask = Image.open(os.path.join(self.mask_dir, f"{image_id}_lesion.bmp")).convert("L")

        if do_flip:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)

        if rot_angle != 0:
            image = TF.rotate(
                image, rot_angle,
                interpolation=InterpolationMode.BILINEAR,
                expand=True,
                fill=0
            )
            if mask is not None:
                mask = TF.rotate(
                    mask, rot_angle,
                    interpolation=InterpolationMode.NEAREST,
                    expand=True,
                    fill=0
                )

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = TF.resize(image, (224, 224), interpolation=InterpolationMode.BILINEAR)
            image = T.ToTensor()(image)
            image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

            if mask is None:
                mask = torch.zeros((1, 224, 224), dtype=image.dtype)
            else:
                mask = TF.resize(mask, (224, 224), interpolation=InterpolationMode.NEAREST)
                mask = T.ToTensor()(mask)
                mask = (mask > 0.5).float()

        aug_image_id = f"{image_id}__{op_name}"
        return image, label, mask, aug_image_id

    def get_label(self, dx):
        label_map = {
            'Melanoma': 0,
            'Atypical Nevus': 1,
            'Common Nevus': 2,
        }
        return label_map.get(dx, 7)

class ISIC2017Dataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir=None, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_id = self.annotations.iloc[idx, 0]
        dx = self.annotations.iloc[idx, 1]
        label = self.get_label(dx)

        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        mask = None
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, f"{image_id}_segmentation.png")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
            else:
                mask = Image.new("L", image.size, 0)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = TF.resize(image, (224,224), interpolation=InterpolationMode.BILINEAR)
            image = T.ToTensor()(image)
            image = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(image)

            if mask is None:
                mask = torch.zeros((1,224,224), dtype=image.dtype)
            else:
                mask = TF.resize(mask, (224,224), interpolation=InterpolationMode.NEAREST)
                mask = T.ToTensor()(mask)
                mask = (mask > 0.5).float()

        return image, label, mask, image_id

    def get_label(self, dx):
        label_map = {'seborrheic_keratosis': 0, 'melanoma': 1, 'other': 2}
        return label_map.get(dx, 7)
class ISIC2018Dataset(Dataset):
    def __init__(self, csv_file, img_dir, mask_dir=None, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_id = self.annotations.iloc[idx, 0]
        dx = self.annotations.iloc[idx, 1]
        label = self.get_label(dx)

        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert("RGB")

        mask = None
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, f"{image_id}_segmentation.png")
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("L")
            else:
                mask = Image.new("L", image.size, 0)

        if self.transform is not None:
            image, mask = self.transform(image, mask)
        else:
            image = TF.resize(image, (224, 224), interpolation=InterpolationMode.BILINEAR)
            image = T.ToTensor()(image)
            image = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)

            if mask is None:
                mask = torch.zeros((1, 224, 224), dtype=image.dtype)
            else:
                mask = TF.resize(mask, (224, 224), interpolation=InterpolationMode.NEAREST)
                mask = T.ToTensor()(mask)
                mask = (mask > 0.5).float()

        return image, label, mask, image_id

    def get_label(self, dx):
        label_map = {'A1_Benign_melanocytic': 0, 'A2_Melanoma': 1, 'A3_Benign_epidermal': 2, 'A4_Other': 3}
        return label_map.get(dx, 7)

class JointTrainTransform:
    def __init__(
        self,
        size=224,
        rrc_scale=(0.8, 1.0),
        rrc_ratio=(3/4, 4/3),
        hflip_p=0.5,
        vflip_p=0.5,
        affine_degrees=15,
        affine_translate=(0.1, 0.1),
        affine_scale=(0.9, 1.1),
        perspective_p=0.5,
        perspective_distortion=0.2,
        color_jitter=(0.2, 0.2, 0.2, 0.1),
        erasing_p=0.5,
        erasing_scale=(0.02, 0.1),
        erasing_ratio=(0.3, 3.3),
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        mask_threshold=True,
    ):
        self.size = (size, size) if isinstance(size, int) else tuple(size)

        self.rrc_scale = rrc_scale
        self.rrc_ratio = rrc_ratio
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p

        self.affine_degrees = (-affine_degrees, affine_degrees) if isinstance(affine_degrees, (int, float)) else affine_degrees
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale

        self.perspective_p = perspective_p
        self.perspective_distortion = perspective_distortion

        b, c, s, h = color_jitter
        self.color_jitter = T.ColorJitter(brightness=b, contrast=c, saturation=s, hue=h)

        self.to_tensor = T.ToTensor()
        self.random_erasing = T.RandomErasing(p=erasing_p, scale=erasing_scale, ratio=erasing_ratio)
        self.normalize = T.Normalize(mean=mean, std=std)

        self.mask_threshold = mask_threshold

    def __call__(self, image, mask):
        if mask is None:
            mask = None

        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        i, j, h, w = T.RandomResizedCrop.get_params(image, scale=self.rrc_scale, ratio=self.rrc_ratio)
        image = TF.resized_crop(image, i, j, h, w, self.size, interpolation=InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.resized_crop(mask, i, j, h, w, self.size, interpolation=InterpolationMode.NEAREST)

        if random.random() < self.hflip_p:
            image = TF.hflip(image)
            if mask is not None:
                mask = TF.hflip(mask)

        if random.random() < self.vflip_p:
            image = TF.vflip(image)
            if mask is not None:
                mask = TF.vflip(mask)

        angle, translations, scale, shear = T.RandomAffine.get_params(
            degrees=self.affine_degrees,
            translate=self.affine_translate,
            scale_ranges=self.affine_scale,
            shears=None,
            img_size=self.size,
        )
        image = TF.affine(
            image, angle=angle, translate=translations, scale=scale, shear=shear,
            interpolation=InterpolationMode.BILINEAR, fill=0
        )
        if mask is not None:
            mask = TF.affine(
                mask, angle=angle, translate=translations, scale=scale, shear=shear,
                interpolation=InterpolationMode.NEAREST, fill=0
            )

        if random.random() < self.perspective_p:
            width, height = self.size[1], self.size[0]
            startpoints, endpoints = T.RandomPerspective.get_params(width, height, self.perspective_distortion)

            image = TF.perspective(
                image, startpoints=startpoints, endpoints=endpoints,
                interpolation=InterpolationMode.BILINEAR, fill=0
            )
            if mask is not None:
                mask = TF.perspective(
                    mask, startpoints=startpoints, endpoints=endpoints,
                    interpolation=InterpolationMode.NEAREST, fill=0
                )

        image = self.color_jitter(image)

        image = self.to_tensor(image)
        if mask is not None:
            mask = self.to_tensor(mask)
            if self.mask_threshold:
                mask = (mask > 0.5).float()

        image = self.random_erasing(image)

        image = self.normalize(image)

        if mask is None:
            mask = torch.zeros((1, self.size[0], self.size[1]), dtype=image.dtype)

        return image, mask
class JointTestTransform:
    def __init__(self, size=224, mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225), mask_threshold=True):
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(mean=mean, std=std)
        self.mask_threshold = mask_threshold

    def __call__(self, image, mask):
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BILINEAR)
        if mask is not None:
            mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        image = self.normalize(self.to_tensor(image))

        if mask is None:
            mask = torch.zeros((1, self.size[0], self.size[1]), dtype=image.dtype)
        else:
            mask = self.to_tensor(mask)
            if self.mask_threshold:
                mask = (mask > 0.5).float()

        return image, mask
train_transform = JointTrainTransform(size=224)
test_transform  = JointTestTransform(size=224)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def normalize01_torch(x: torch.Tensor, eps=1e-6):
    if x.dim() == 3:
        x_ = x.view(x.size(0), -1)
        mn = x_.min(dim=1, keepdim=True)[0].view(-1, 1, 1)
        mx = x_.max(dim=1, keepdim=True)[0].view(-1, 1, 1)
        return (x - mn) / (mx - mn + eps)
    else:
        x_ = x.view(x.size(0), -1)
        mn = x_.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        mx = x_.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        return (x - mn) / (mx - mn + eps)

class LocalSpectralDescriptor(nn.Module):
    def __init__(self, patch_size=16, num_bands=3, band_edges=(0.25, 0.55), eps=1e-6,
                 imagenet_mean=(0.485, 0.456, 0.406), imagenet_std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.patch_size = patch_size
        self.num_bands = num_bands
        self.band_edges = band_edges
        self.eps = eps

        mean = torch.tensor(imagenet_mean).view(1, 3, 1, 1)
        std = torch.tensor(imagenet_std).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        self.register_buffer("band_masks", self._build_band_masks(patch_size, num_bands, band_edges))

    def _build_band_masks(self, P, d, band_edges):
        fy = torch.fft.fftfreq(P) * P
        fx = torch.fft.rfftfreq(P) * P
        grid_y, grid_x = torch.meshgrid(fy, fx, indexing='ij')
        radius = torch.sqrt(grid_y ** 2 + grid_x ** 2)
        radius = radius / (radius.max() + 1e-12)

        edges = list(band_edges)
        if d != 3:
            raise ValueError("Only num_bands=3 is currently implemented.")

        e1, e2 = edges[0], edges[1]
        m0 = (radius <= e1).float()
        m1 = ((radius > e1) & (radius <= e2)).float()
        m2 = (radius > e2).float()
        masks = torch.stack([m0, m1, m2], dim=0)
        return masks

    def forward(self, img, H_feat, W_feat):
        B, _, H0, W0 = img.shape

        x = img * self.std + self.mean
        lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        target_h = H_feat * self.patch_size
        target_w = W_feat * self.patch_size
        if (H0, W0) != (target_h, target_w):
            lum = F.interpolate(lum, size=(target_h, target_w), mode='bilinear', align_corners=False)

        P = self.patch_size

        lum = lum.view(B, 1, H_feat, P, W_feat, P)
        patches = lum.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, H_feat * W_feat, P, P)

        patches = patches - patches.mean(dim=(-2, -1), keepdim=True)
        patches = patches.contiguous()
        Fuv = torch.fft.rfft2(patches, dim=(-2, -1), norm='ortho')
        power = (Fuv.real ** 2 + Fuv.imag ** 2)

        masks = self.band_masks.to(power.dtype).unsqueeze(0).unsqueeze(0)
        band_energy = (power.unsqueeze(2) * masks).sum(dim=(-2, -1))
        total = band_energy.sum(dim=-1, keepdim=True) + self.eps
        s = band_energy / total

        S_map = s.permute(0, 2, 1).contiguous().view(B, self.num_bands, H_feat, W_feat)
        return S_map

class CustomResNet34_feature(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet34(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        pooled = self.avgpool(c4)
        cls_output = self.fc(torch.flatten(pooled, 1))
        return cls_output, {"c1": c1, "c2": c2, "c3": c3, "c4": c4}

class SegHeadFPN(nn.Module):
    def __init__(self, c1=64, c2=128, c3=256, c4=512, mid=128, out_ch=1):
        super().__init__()
        self.lat4 = nn.Conv2d(c4, mid, 1)
        self.lat3 = nn.Conv2d(c3, mid, 1)
        self.lat2 = nn.Conv2d(c2, mid, 1)
        self.lat1 = nn.Conv2d(c1, mid, 1)

        self.ref3 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(True)
        )
        self.ref2 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(True)
        )
        self.ref1 = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(True)
        )

        self.head = nn.Sequential(
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(True),
            nn.Conv2d(mid, out_ch, 1)
        )

    def forward(self, f3, c4, c2, c1, out_size):
        p4 = self.lat4(c4)

        p3 = self.lat3(f3) + F.interpolate(p4, size=f3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.ref3(p3)

        p2 = self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.ref2(p2)

        p1 = self.lat1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.ref1(p1)

        logit = self.head(p1)
        logit = F.interpolate(logit, size=out_size, mode="bilinear", align_corners=False)
        return logit

class GNNClusteringLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_clusters):
        super(GNNClusteringLayer, self).__init__()
        self.gnn_embed = DenseSAGEConv(in_channels, hidden_channels)
        self.gnn_pool = DenseSAGEConv(in_channels, num_clusters)

    def forward(self, x, adj, mask=None):
        z = F.relu(self.gnn_embed(x, adj, mask))
        s = F.softmax(self.gnn_pool(x, adj, mask), dim=-1)
        x_coarse, adj_coarse, link_loss, ent_loss = dense_diff_pool(z, adj, s, mask)
        return x_coarse, ent_loss, s


class IntegratedSkinNet(nn.Module):
    def __init__(self, num_classes, num_clusters=5,
                 use_spectral=True, use_gnn=True, use_filter=True,
                 beta_scale=1.0):
        super().__init__()
        self.backbone = CustomResNet34_feature(num_classes)

        self.gnn_input_dim = 256
        self.gnn_hidden_dim = 128
        self.num_clusters = num_clusters
        self.num_classes = num_classes

        self.use_spectral = bool(use_spectral)
        self.use_gnn = bool(use_gnn)
        self.use_filter = bool(use_filter)

        self.pos_fusion = nn.Sequential(
            nn.Conv2d(256 + 2, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.spectral_extractor = LocalSpectralDescriptor(patch_size=16, num_bands=3)
        self.film = nn.Conv2d(3, 2 * self.gnn_input_dim, kernel_size=1, bias=True)
        self.beta_scale = float(beta_scale)

        self.aux_pool = nn.AdaptiveAvgPool2d(1)
        self.aux_fc = nn.Linear(self.gnn_input_dim, num_classes)

        self.gnn_cluster = GNNClusteringLayer(self.gnn_input_dim, self.gnn_hidden_dim, num_clusters)

        self.seg_head = SegHeadFPN(c1=64, c2=128, c3=256, c4=512, mid=128, out_ch=1)

    def set_ablation(self, use_spectral=None, use_gnn=None, use_filter=None):
        if use_spectral is not None: self.use_spectral = bool(use_spectral)
        if use_gnn is not None: self.use_gnn = bool(use_gnn)
        if use_filter is not None: self.use_filter = bool(use_filter)

    def _get_coordinate_grid(self, x):
        B, C, H, W = x.shape
        y = torch.linspace(-1, 1, H, device=x.device)
        x_ = torch.linspace(-1, 1, W, device=x.device)
        gy, gx = torch.meshgrid(y, x_, indexing='ij')
        return torch.stack([gy, gx], dim=0).unsqueeze(0).expand(B, 2, H, W)

    def forward(self, img):
        cls_output, feats = self.backbone(img)
        c1, c2, c3, c4 = feats["c1"], feats["c2"], feats["c3"], feats["c4"]

        pos_grid = self._get_coordinate_grid(c3)
        f3 = self.pos_fusion(torch.cat([c3, pos_grid], dim=1))

        B, C, H, W = f3.shape
        N = H * W

        if self.use_spectral:
            S_map = self.spectral_extractor(img, H, W)
            gb = self.film(S_map)
            gamma, beta = torch.split(gb, C, dim=1)
            gamma = torch.sigmoid(gamma)
            beta = torch.tanh(beta) * self.beta_scale
            feature_map_mod = f3 * (1.0 + gamma) + beta
        else:
            feature_map_mod = f3

        aux_output = self.aux_fc(torch.flatten(self.aux_pool(feature_map_mod), 1))

        seg_logits = self.seg_head(feature_map_mod, c4, c2, c1, out_size=img.shape[-2:])

        if self.use_gnn:
            x_graph = feature_map_mod.view(B, C, -1).permute(0, 2, 1)

            x_norm = F.normalize(x_graph, p=2, dim=-1)
            adj_feature = torch.bmm(x_norm, x_norm.transpose(1, 2))

            y_coord = torch.linspace(0, 1, H, device=img.device)
            x_coord = torch.linspace(0, 1, W, device=img.device)
            gy, gx = torch.meshgrid(y_coord, x_coord, indexing='ij')
            coords = torch.stack([gx.flatten(), gy.flatten()], dim=1)
            dist = torch.cdist(coords, coords, p=2).unsqueeze(0).expand(B, -1, -1)

            sigma = 0.5
            spatial_weight = torch.exp(- (dist ** 2) / (2 * sigma ** 2))
            adj = torch.relu(adj_feature) * spatial_weight

            centers, e_loss, assignments = self.gnn_cluster(x_graph, adj)

            centers_norm = F.normalize(centers, p=2, dim=-1)
            center_sim = torch.bmm(centers_norm, centers_norm.transpose(1, 2))
            eye = torch.eye(self.num_clusters, device=img.device).unsqueeze(0).expand(B, -1, -1)
            ortho_loss = torch.mean(torch.sum((center_sim * (1 - eye)) ** 2, dim=[1, 2]))

        else:
            assignments = torch.full((B, N, self.num_clusters),
                                     1.0 / float(self.num_clusters),
                                     device=img.device, dtype=feature_map_mod.dtype)
            centers = torch.zeros((B, self.num_clusters, self.gnn_hidden_dim),
                                  device=img.device, dtype=feature_map_mod.dtype)
            l_loss = torch.zeros((), device=img.device, dtype=feature_map_mod.dtype)
            e_loss = torch.zeros((), device=img.device, dtype=feature_map_mod.dtype)
            ortho_loss = torch.zeros((), device=img.device, dtype=feature_map_mod.dtype)

        return (cls_output, centers, assignments, e_loss,
                feature_map_mod, ortho_loss, c3, aux_output, seg_logits)
def dice_loss_with_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1,2,3))
    den = (prob + target).sum(dim=(1,2,3)) + eps
    return 1 - (num / den).mean()


@torch.no_grad()
def batch_dice_iou(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-6):
    B = pred_bin.shape[0]
    pred = pred_bin.reshape(B, -1).float()
    gt = gt_bin.reshape(B, -1).float()

    inter = (pred * gt).sum(dim=1)
    pred_sum = pred.sum(dim=1)
    gt_sum = gt.sum(dim=1)

    dice = (2.0 * inter + eps) / (pred_sum + gt_sum + eps)
    union = pred_sum + gt_sum - inter
    iou = (inter + eps) / (union + eps)
    return dice, iou

class LayerCAMExtractor:

        def __init__(self, target_layer: nn.Module):
            self.target_layer = target_layer
            self.act = None
            self.hook = target_layer.register_forward_hook(self._forward_hook)

        def _forward_hook(self, module, inp, out):
            self.act = out
            self.act.retain_grad()

        @torch.no_grad()
        def close(self):
            if self.hook is not None:
                self.hook.remove()
                self.hook = None

        def compute(self, logit: torch.Tensor, label: torch.Tensor, retain_graph=True, eps=1e-6):
            assert self.act is not None, "No activation captured. Make sure a forward pass happened."

            B, C, h, w = self.act.shape


            score = logit.float().gather(1, label.view(-1, 1)).sum()

            grads = torch.autograd.grad(
                outputs=score,
                inputs=self.act,
                retain_graph=retain_graph,
                create_graph=False,
                only_inputs=True
            )[0]


            cam = F.relu((F.relu(grads) * self.act).sum(dim=1, keepdim=True))

            cam_flat = cam.view(B, -1)
            cam_min = cam_flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            cam_max = cam_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
            cam = (cam - cam_min) / (cam_max - cam_min + eps)

            return cam
def dice_iou_from_binary(pred_bhw: torch.Tensor, gt_bhw: torch.Tensor, eps=1e-6):
        pred = pred_bhw.float()
        gt = gt_bhw.float()
        inter = (pred * gt).sum(dim=(1, 2))
        union = (pred + gt - pred * gt).sum(dim=(1, 2))
        dice = (2 * inter + eps) / (pred.sum(dim=(1, 2)) + gt.sum(dim=(1, 2)) + eps)
        iou = (inter + eps) / (union + eps)
        return dice.mean().item(), iou.mean().item()
def train(current_epoch, train_loader, model, optimizer, scaler, args, writer, loss_func):
    use_spectral = bool(getattr(args, "use_spectral", True))
    use_gnn = bool(getattr(args, "use_gnn", True))
    use_filter = bool(getattr(args, "use_filter", True))

    if hasattr(model, "set_ablation"):
        model.set_ablation(use_spectral=use_spectral, use_gnn=use_gnn, use_filter=use_filter)

    device = getattr(args, "device", "cuda")

    if use_filter and (not hasattr(model, "_gf_rgb")):
        model._gf_rgb = SideWindowMeanFilter().to(device)
        model._gf_rgb.eval()

    if not hasattr(model, "_layercam_extractor"):
        target_layer = model.backbone.layer3[0]
        model._layercam_extractor = LayerCAMExtractor(target_layer)

    
    gnn_start_epoch = 20

    is_warmup = current_epoch < gnn_start_epoch
    print(f"training...... [Phase: {'Backbone Warmup' if is_warmup else 'Joint Training'}]")

    
    for _, param in model.named_parameters():
        param.requires_grad = True

    frozen_params = [name for name, p in model.named_parameters() if not p.requires_grad]
    print(f"[Epoch {current_epoch}] Frozen parameters count: {len(frozen_params)}")

    model.train()

    global_counter = args.global_counter
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()

    torch.cuda.empty_cache()
    batch_counter = 0

    if (not use_gnn) or is_warmup:
        cluster_kmeans_weight = 0.0
        cluster_struct_weight = 0.0
        ortho_loss_weight = 0.0
    else:
        dataset_name = getattr(args, "dataset_name", "ISIC2018")
        cluster_kmeans_weight_map = {
            "ISIC2018": 0.08,
            "ISIC2017": 0.05,
            "PH2": 0.03,
        }
        if dataset_name not in cluster_kmeans_weight_map:
            raise ValueError(f"Unsupported dataset for cluster_kmeans_weight: {dataset_name}")

        cluster_kmeans_weight = cluster_kmeans_weight_map[dataset_name]
        cluster_struct_weight = 0.01
        ortho_loss_weight = 0.1
    print(f"[Epoch {current_epoch}] cluster_kmeans_weight={cluster_kmeans_weight}")

    seg_delay = int(getattr(args, "seg_delay", 20))
    seg_start_epoch = gnn_start_epoch + seg_delay

    for data in train_loader:
        batch_counter += 1
        print(f"Processing batch {batch_counter}/{len(train_loader)}", end="\r")

        img, label = data[0], data[1]

        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        use_seg = (current_epoch >= seg_start_epoch)

        with autocast(device_type='cuda', enabled=True):
            outputs = model(img)
            logit, centers, assignments, e_loss, feature_map, ortho_loss, _, aux_logit, seg_logits = outputs

            cls_loss = loss_func(logit, label.long())
            aux_loss = loss_func(aux_logit, label.long())

            aux_loss_weight = 0.0
            if use_spectral:
                aux_loss_weight = 0.5 if is_warmup else 0.2

            if use_gnn:
                B, C, H, W = feature_map.shape
                S = assignments

                x_graph = feature_map.view(B, C, -1).permute(0, 2, 1).contiguous()
                x_graph = F.normalize(x_graph, p=2, dim=2)

                eps = 1e-6
                S_sum = S.sum(dim=1).unsqueeze(-1)
                mu = torch.bmm(S.transpose(1, 2), x_graph) / (S_sum + eps)
                mu = F.normalize(mu, p=2, dim=2)

                dot = torch.bmm(x_graph, mu.permute(0, 2, 1))
                dists = 2.0 - 2.0 * dot

                cluster_kmeans_loss = (S * dists).sum(dim=(1, 2)).mean()

                cluster_struct_loss =e_loss

                total_cluster_loss = (cluster_kmeans_weight * cluster_kmeans_loss) + \
                                     (cluster_struct_weight * cluster_struct_loss)
                
                total_ortho_loss = ortho_loss_weight * ortho_loss
            else:
                cluster_kmeans_loss = torch.zeros((), device=img.device)
                cluster_struct_loss = torch.zeros((), device=img.device)
                total_cluster_loss = torch.zeros((), device=img.device)
                total_ortho_loss = torch.zeros((), device=img.device)

        if use_seg:
            with torch.cuda.amp.autocast(enabled=False):
                layercam_feat = model._layercam_extractor.compute(
                    logit=logit.float(),
                    label=label.long(),
                    retain_graph=True
                )

                model.zero_grad(set_to_none=True)

                B2, _, h, w = layercam_feat.shape

                if use_gnn:
                    K = assignments.shape[-1]
                    assign_map = assignments.float().permute(0, 2, 1).contiguous().view(B2, K, h, w)

                    thr = 0.3
                    cam_bin = (layercam_feat > thr)
                    ass_bin = (assign_map > thr)

                    inter = (ass_bin & cam_bin).sum(dim=(2, 3)).float()
                    union = (ass_bin | cam_bin).sum(dim=(2, 3)).float()
                    iou = inter / (union + 1e-6)

                    best_iou, best_k = iou.max(dim=1)
                    valid = (best_iou > 0.3)

                    best_assign = assign_map[torch.arange(B2, device=assign_map.device), best_k].unsqueeze(1)
                    fused_out = 0.5 * (layercam_feat + best_assign)
                    fused_out = torch.where(valid.view(B2, 1, 1, 1), fused_out, layercam_feat)
                else:
                    fused_out = layercam_feat

                fused_flat = fused_out.view(B2, -1)
                fmin = fused_flat.min(dim=1, keepdim=True)[0].view(B2, 1, 1, 1)
                fmax = fused_flat.max(dim=1, keepdim=True)[0].view(B2, 1, 1, 1)
                fused_out = (fused_out - fmin) / (fmax - fmin + 1e-6)

                fused_up = F.interpolate(
                    fused_out, size=(img.shape[2], img.shape[3]),
                    mode='bilinear', align_corners=False
                ).float()

            with torch.no_grad():
                if use_filter:
                    img01 = denorm_to_01(img).float()
                    p_in = fused_up.detach().float()
                    fused_gf = model._gf_rgb(img01, p_in)
                    fused_gf = normalize01_torch(fused_gf).clamp(0, 1)
                else:
                    fused_gf = normalize01_torch(fused_up.detach().float()).clamp(0, 1)

                pseudo_mode = getattr(args, "pseudo_mode", "soft")
                pseudo_thr = float(getattr(args, "pseudo_thr", 0.5))

                if pseudo_mode.lower() in ["hard", "binary", "bin"]:
                    pseudo_target = (fused_gf > pseudo_thr).float()
                else:
                    pseudo_target = fused_gf

            seg_logits_f = seg_logits.float()
            seg_bce = F.binary_cross_entropy_with_logits(seg_logits_f, pseudo_target)
            seg_dice = dice_loss_with_logits(seg_logits_f, pseudo_target)
            seg_loss = seg_bce + seg_dice
            seg_w = float(getattr(args, "seg_w", 1.0))
        else:
            seg_loss = torch.zeros((), device=img.device)
            seg_w = 0.0

        total_loss = cls_loss + aux_loss_weight * aux_loss + total_cluster_loss + total_ortho_loss + seg_w * seg_loss

        optimizer.zero_grad()
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        logit_argmax = torch.argmax(logit, dim=1).detach().cpu().numpy()
        label_np = label.detach().cpu().numpy()

        train_loss.update(total_loss.item(), img.size(0))
        accuracy = accuracy_score(label_np, logit_argmax)
        train_accuracy.update(accuracy, img.size(0))

        global_counter += 1

        if writer and global_counter % args.show_interval == 0:
            writer.add_scalar('train/loss', train_loss.avg, global_counter)
            writer.add_scalar('train/acc', train_accuracy.avg, global_counter)
            writer.add_scalar('train/cls_loss', cls_loss.item(), global_counter)

            writer.add_scalar('train/cluster_kmeans_loss', float(cluster_kmeans_loss.item()), global_counter)
            writer.add_scalar('train/cluster_struct_loss', float(cluster_struct_loss.item()), global_counter)
            writer.add_scalar('train/ortho_loss', float(ortho_loss.item()), global_counter)

            if use_seg:
                writer.add_scalar('train/seg_loss', float(seg_loss.item()), global_counter)

    args.global_counter = global_counter

    print('\n')

    logging.debug(
        "epoch {}, train loss {:.4f}, accuracy {:.4f}".format(
            current_epoch, train_loss.avg, train_accuracy.avg
        )
    )
    print(
        "epoch {}, train loss {:.4f}, accuracy {:.4f}".format(
            current_epoch, train_loss.avg, train_accuracy.avg
        )
    )

    return {
        "train_loss": train_loss.avg,
        "accuracy": train_accuracy.avg,
        "cluster_kmeans_weight": cluster_kmeans_weight,
        "cluster_struct_weight": cluster_struct_weight,
    }


def validate(current_epoch, test_loader, model, loss_func, args, writer):
    print('\nvalidating ... ', flush=True, end='')
    core = model
    save_seg224 = bool(getattr(args, "save_seg224", False))
    seg_save_root = getattr(args, "seg_save_root", "./seg_outputs")
    dataset_name = getattr(args, "dataset_name", "default_dataset")
    save_cam_vis = bool(getattr(args, "save_cam_vis", False))
    cam_cmap_name = getattr(args, "cam_cmap", "jet")
    cam_alpha = float(getattr(args, "cam_alpha", 0.45))

    def _get_weight_dir(_args, _writer=None):
        cand_keys = [
            "resume", "weights", "weight_path", "ckpt", "ckpt_path",
            "checkpoint", "model_path", "pretrained", "eval_ckpt"
        ]
        for k in cand_keys:
            p = getattr(_args, k, None)
            if not p:
                continue
            if isinstance(p, (list, tuple)) and len(p) > 0:
                p = p[0]
            if isinstance(p, str):
                p = os.path.expanduser(p)
                if os.path.isdir(p):
                    return p
                return os.path.dirname(p)
        if _writer is not None and hasattr(_writer, "log_dir") and _writer.log_dir:
            return _writer.log_dir
        return "./"

    weight_dir = _get_weight_dir(args, writer)
    cam_dir = os.path.join(weight_dir, "cam_vis", dataset_name)

    if save_cam_vis:
        os.makedirs(cam_dir, exist_ok=True)

    _cmap = cm.get_cmap(cam_cmap_name)

    def _save_cam_pair(img01_chw: torch.Tensor, cam01_hw: torch.Tensor,
                       base_name: str, dice_val=None, iou_val=None, tag: str = "base"):
        img = (img01_chw.detach().clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
        cam01 = cam01_hw.detach().clamp(0, 1).cpu().numpy()

        if dice_val is None or iou_val is None:
            suf = f"_diceNA_iouNA_{tag}"
        else:
            suf = f"_dice{dice_val:.4f}_iou{iou_val:.4f}_{tag}"


        heat = (_cmap(cam01)[..., :3] * 255.0).astype(np.uint8)
        overlay = (img.astype(np.float32) * (1.0 - cam_alpha) + heat.astype(np.float32) * cam_alpha)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        Image.fromarray(overlay, mode="RGB").save(os.path.join(cam_dir, f"{base_name}_cam_overlay{suf}.png"))

    our_dir = os.path.join(seg_save_root, dataset_name, "seg224_our")
    gt_dir = os.path.join(seg_save_root, dataset_name, "gt224")

    if save_seg224:
        os.makedirs(our_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

    def _basename_no_ext(x):
        if isinstance(x, str):
            return os.path.splitext(os.path.basename(x))[0]
        if hasattr(x, "__fspath__"):
            p = os.fspath(x)
            return os.path.splitext(os.path.basename(p))[0]
        return str(x)

    @torch.no_grad()
    def _dice_iou_vec(pred_bin: torch.Tensor, gt_bin: torch.Tensor, eps: float = 1e-6):
        pb = pred_bin.flatten(1)
        gb = gt_bin.flatten(1)

        inter = (pb & gb).sum(1).float()
        ps = pb.sum(1).float()
        gs = gb.sum(1).float()
        uni = (pb | gb).sum(1).float()

        dice = (2 * inter) / (ps + gs + eps)
        iou = inter / (uni + eps)

        empty = (ps + gs) == 0
        dice = torch.where(empty, torch.ones_like(dice), dice)
        iou = torch.where(empty, torch.ones_like(iou), iou)
        return dice, iou

    sample_idx = 0

    use_spectral = bool(getattr(args, "use_spectral", True))
    use_gnn      = bool(getattr(args, "use_gnn", True))
    use_filter   = bool(getattr(args, "use_filter", False))

    def _ablation_tag(use_spectral: bool, use_gnn: bool, use_filter: bool) -> str:
        if not (use_spectral or use_gnn or use_filter):
            return "base"
        if use_filter:
            return "filter"
        if use_spectral:
            return "spectral"
        if use_gnn:
            return "gnn_w_oQ"
        return "base"

    _ab_tag = _ablation_tag(use_spectral, use_gnn, use_filter)

    if hasattr(core, "set_ablation"):
        core.set_ablation(use_spectral=use_spectral, use_gnn=use_gnn, use_filter=use_filter)

    if use_filter and (not hasattr(core, "_ac_refiner")):
        core._ac_refiner = SideWindowMeanFilter().to(device)
        core._ac_refiner.eval()

    model.eval()
    val_loss = AverageMeter()
    val_label, val_logit = [], []

    if not hasattr(core, "_layercam_extractor"):
        target_layer = core.backbone.layer3[0]
        core._layercam_extractor = LayerCAMExtractor(target_layer)

    dice_sum, iou_sum, seg_cnt = 0.0, 0.0, 0
    seg_head_dice_sum, seg_head_iou_sum, seg_head_cnt = 0.0, 0.0, 0

    for data in test_loader:
        img, label = data[0], data[1]
        mask = data[2] if len(data) >= 3 else None

        img = img.to('cuda', non_blocking=True)
        label = label.to('cuda', non_blocking=True)

        if mask is not None and torch.is_tensor(mask):
            mask = mask.to('cuda', non_blocking=True)
            if mask.dim() == 4 and mask.size(1) == 1:
                gt = (mask[:, 0] > 0.5)
            elif mask.dim() == 3:
                gt = (mask > 0.5)
            else:
                gt = (mask.squeeze() > 0.5)
        else:
            gt = None

        with autocast(device_type='cuda', enabled=True):
            outputs = model(img)
            logit, centers, assignments, _, _, _, _, _, seg_logits = outputs
            loss = loss_func(logit, label.long())

        with torch.cuda.amp.autocast(enabled=False):
            layercam_feat = core._layercam_extractor.compute(
                logit=logit.float(),
                label=label.long(),
                retain_graph=False
            )

            B, _, h, w = layercam_feat.shape

            if use_gnn:
                K = assignments.shape[-1]
                assign_map = assignments.float().permute(0, 2, 1).contiguous().view(B, K, h, w)

                thr_sel = 0.3
                cam_bin = (layercam_feat > thr_sel)
                ass_bin = (assign_map > thr_sel)

                inter = (ass_bin & cam_bin).sum(dim=(2,3)).float()
                union = (ass_bin | cam_bin).sum(dim=(2,3)).float()
                iou_mat = inter / (union + 1e-6)

                best_iou, best_k = iou_mat.max(dim=1)
                valid = (best_iou > 0.3)

                best_assign = assign_map[torch.arange(B, device=assign_map.device), best_k].unsqueeze(1)
                fused = 0.5 * (layercam_feat + best_assign)
                fused = torch.where(valid.view(B,1,1,1), fused, layercam_feat)
            else:
                fused = layercam_feat

            fused_flat = fused.view(B, -1)
            fmin = fused_flat.min(dim=1, keepdim=True)[0].view(B,1,1,1)
            fmax = fused_flat.max(dim=1, keepdim=True)[0].view(B,1,1,1)
            fused = (fused - fmin) / (fmax - fmin + 1e-6)

            fused_up = F.interpolate(
                fused, size=(img.shape[2], img.shape[3]),
                mode='bilinear', align_corners=False
            ).float()

        with torch.no_grad():
            img01 = denorm_to_01(img).float()
            if use_filter:
                fused_ac = core._ac_refiner(img01, fused_up.detach().float())
                fused_ac = normalize01_torch(fused_ac).clamp(0, 1)
            else:
                fused_ac = normalize01_torch(fused_up.detach().float()).clamp(0, 1)
            if save_cam_vis:
                B = fused_ac.shape[0]

                if len(data) >= 4:
                    names = data[3]
                    if isinstance(names, (list, tuple)):
                        names = [_basename_no_ext(n) for n in names]
                    else:
                        names = [_basename_no_ext(names) for _ in range(B)]
                else:
                    names = [f"{sample_idx + i:06d}" for i in range(B)]
                    sample_idx += B

                if gt is not None:
                    pred_bin = (fused_ac[:, 0] > 0.5)
                    dice_vec, iou_vec = _dice_iou_vec(pred_bin, gt)
                else:
                    dice_vec, iou_vec = None, None

                for i in range(B):
                    if gt is not None:
                        d = float(dice_vec[i].item())
                        u = float(iou_vec[i].item())
                    else:
                        d, u = None, None

                    _save_cam_pair(
                        img01[i],
                        fused_ac[i, 0],
                        names[i],
                        dice_val=d,
                        iou_val=u,
                        tag=_ab_tag
                    )
        if gt is not None:
            pred_bin = (fused_ac[:, 0] > 0.5)
            dice_vec, iou_vec = _dice_iou_vec(pred_bin, gt)  # [B], [B]

            dice_sum += float(dice_vec.sum().item())
            iou_sum += float(iou_vec.sum().item())
            seg_cnt += int(dice_vec.numel())

        if gt is not None:
            with torch.no_grad():
                seg_prob = torch.sigmoid(seg_logits.float())
                seg_pred_bin = (seg_prob[:, 0] > 0.5)

            dice_h_vec, iou_h_vec = _dice_iou_vec(seg_pred_bin, gt)  # [B], [B]

            seg_head_dice_sum += float(dice_h_vec.sum().item())
            seg_head_iou_sum += float(iou_h_vec.sum().item())
            seg_head_cnt += int(dice_h_vec.numel())
        if save_seg224:
            B = seg_prob.shape[0]

            if len(data) >= 4:
                names = data[3]
                if isinstance(names, (list, tuple)):
                    names = [_basename_no_ext(n) for n in names]
                else:
                    names = [_basename_no_ext(names) for _ in range(B)]
            else:
                names = [f"{sample_idx + i:06d}" for i in range(B)]
            sample_idx += B

            seg_prob_224 = F.interpolate(seg_prob, size=(224, 224), mode='bilinear', align_corners=False)
            seg_bin_224 = (seg_prob_224[:, 0] > 0.5)

            gt_224 = F.interpolate(
                gt.float().unsqueeze(1), size=(224, 224), mode='nearest'
            )[:, 0] > 0.5

            dice_vec, iou_vec = _dice_iou_vec(seg_bin_224, gt_224)

            seg_u8 = (seg_bin_224.to(torch.uint8) * 255).cpu().numpy()
            gt_u8 = (gt_224.to(torch.uint8) * 255).cpu().numpy()

            for i in range(B):
                d = float(dice_vec[i].item())
                u = float(iou_vec[i].item())

                out_name = f"{names[i]}_dice{d:.4f}_iou{u:.4f}_our.png"
                gt_name = f"{names[i]}.png"

                Image.fromarray(seg_u8[i], mode='L').save(os.path.join(our_dir, out_name))
                Image.fromarray(gt_u8[i], mode='L').save(os.path.join(gt_dir, gt_name))
        logit_argmax = torch.argmax(logit, dim=1)
        val_label.extend(label.detach().cpu().tolist())
        val_logit.extend(logit_argmax.detach().cpu().tolist())
        val_loss.update(loss.item(), img.size(0))

    cls_f1 = f1_score(val_label, val_logit, average='weighted', zero_division=0)

    pseudo_dice = dice_sum / max(seg_cnt, 1)
    pseudo_iou  = iou_sum  / max(seg_cnt, 1)

    seg_head_dice = seg_head_dice_sum / max(seg_head_cnt, 1)
    seg_head_iou  = seg_head_iou_sum  / max(seg_head_cnt, 1)

    print(
        f"valid loss {val_loss.avg:.4f}, cls_f1 {cls_f1:.4f}, "
        f"pseudo_dice {pseudo_dice:.4f}, pseudo_iou {pseudo_iou:.4f}, "
        f"seg_dice {seg_head_dice:.4f}, seg_iou {seg_head_iou:.4f}\n"
    )

    return cls_f1, pseudo_dice, pseudo_iou, seg_head_dice, seg_head_iou



if __name__ == '__main__':
    args = config()

    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    save_path = os.path.join(args.save_folder, f"{args.dataset_name}_resnet34_{nowTime}")
    os.makedirs(save_path, exist_ok=True)

    class Tee:

        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()



    log_file_path = os.path.join(save_path, "train.log")
    log_f = open(log_file_path, "a", buffering=1, encoding="utf-8")

    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)




    if args.dataset_name == 'PH2':
        train_csv_file = "../datasets/PH2/Data_enhancement_train.csv"
        test_csv_file = "../datasets/PH2/PH2_test_transformed.csv"
        img_dir = "../datasets/PH2/Data_enhancement_train"
        test_dir = '../datasets/PH2/image'
        train_mask_dir = '../datasets/PH2/Data_enhancement_mask_train'
        test_mask_dir = '../datasets/PH2/mask'
        train_dataset = PH2Dataset(csv_file=train_csv_file, img_dir=img_dir, mask_dir=train_mask_dir, transform=train_transform)
        test_dataset = PH2Dataset(csv_file=test_csv_file, img_dir=test_dir, mask_dir=test_mask_dir, transform=test_transform)
    elif args.dataset_name == 'ISIC2017':
        train_csv_file = "../datasets/ISIC2017/train_set.csv"
        test_csv_file = "../datasets/ISIC2017/test_set.csv"
        img_dir = "../datasets/ISIC2017/ISIC-2017_Training_Data"
        test_dir = '../datasets/ISIC2017/ISIC-2017_Training_Data'
        mask_dir = "../datasets/ISIC2017/ISIC-2017_Training_Part1_GroundTruth"
        train_dataset = ISIC2017Dataset(csv_file=train_csv_file, img_dir=img_dir, mask_dir=mask_dir, transform=train_transform)
        test_dataset = ISIC2017Dataset(csv_file=test_csv_file, img_dir=test_dir, mask_dir=mask_dir, transform=test_transform)
    elif args.dataset_name == 'ISIC2018':
        train_csv_file = "../datasets/ISIC2018/isic_schemeA_train_7_3.csv"
        test_csv_file = "../datasets/ISIC2018/isic_schemeA_test_7_3.csv"
        img_dir = "../datasets/ISIC2018/ISIC2018_image"
        test_dir = '../datasets/ISIC2018/ISIC2018_image'
        mask_dir = "../datasets/ISIC2018/ISIC2018_GT"
        train_dataset = ISIC2018Dataset(csv_file=train_csv_file, img_dir=img_dir, mask_dir=mask_dir, transform=train_transform)
        test_dataset = ISIC2018Dataset(csv_file=test_csv_file, img_dir=test_dir, mask_dir=mask_dir, transform=test_transform)
    else:
        print('NO DATASET')

    train_loader = DataLoader(
        train_dataset,
        batch_size=config().batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        persistent_workers=True,
        prefetch_factor=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config().batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        persistent_workers=True,
        worker_init_fn = worker_init_fn
    )
    model, optimizer, scheduler, scaler = get_model(args)


    

    print('Running parameters:\n', args)
    print('# of train dataset:', len(train_loader) * args.batch_size)
    print('# of valid dataset:', len(test_loader) * args.batch_size)
    print()

    logging.debug('Running parameters: {}'.format(args))
    logging.debug('train dataset: {}'.format(len(train_loader) * args.batch_size))
    logging.debug('valid dataset: {}'.format(len(test_loader) * args.batch_size))
    logging.debug("")

    loss_func = nn.CrossEntropyLoss()

    if getattr(args, "eval_ckpt", ""):
        ckpt_path = os.path.expanduser(args.eval_ckpt)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--eval_ckpt not found: {ckpt_path}")

        print(f"\n[EVAL_ONCE] Loading checkpoint: {ckpt_path}\n")
        model = load_checkpoint_to_model(
            model,
            ckpt_path,
            device=device,
            strict=bool(getattr(args, "eval_strict", False)),
        )

        cls_f1, pseudo_dice, pseudo_iou, seg_dice, seg_iou = validate(
            0, test_loader, model, loss_func, args, writer=None
        )

        print(
            "[EVAL_ONCE] "
            f"cls_f1={cls_f1:.4f}, "
            f"pseudo_dice={pseudo_dice:.4f}, pseudo_iou={pseudo_iou:.4f}, "
            f"seg_dice={seg_dice:.4f}, seg_iou={seg_iou:.4f}"
        )

        if getattr(args, "eval_only", False):
            print("[EVAL_ONCE] eval_only=True, exit before training.\n")
            raise SystemExit(0)

    with tqdm(total=args.epoch, desc='Training Progress') as pbar:
        for current_epoch in range(1, args.epoch + 1):
            start_time = time.time()
            train(current_epoch, train_loader, model, optimizer, scaler, args, writer=None, loss_func=loss_func)
            end_time = time.time()
            print(end_time - start_time)

            scheduler.step()

            state = {
                'model': model.state_dict(),
            }

            if current_epoch % 3 == 0:
                model_file = os.path.join(save_path, f"epoch_{current_epoch}.pth")
                torch.save(state, model_file)
                print(f"\nSaving checkpoint every 3 epochs: {model_file}\n")
                logging.debug(f"\nSaving checkpoint every 3 epochs: {model_file}\n")

            pbar.set_description(f"Epoch {current_epoch}/{args.epoch}")
            pbar.update(1)

