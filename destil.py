# %%

# %% [markdown]
# # Librerías

# %%
import torch
import wandb
import timm
import cv2
import geffnet
import time
import random
import argparse
import albumentations
import os
import math
import sys
import PIL.Image
from resnest.torch import resnest101
from pretrainedmodels import se_resnext101_32x4d
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.optim.optim_factory import create_optimizer
from timm.utils import NativeScaler, accuracy
from types import SimpleNamespace
from typing import Iterable, Optional
from deit.loss import DistillationLoss
from deit.utils import MetricLogger, SmoothedValue
from torch.utils.tensorboard import SummaryWriter



args = SimpleNamespace()

# Experimento
args.experiment = "EffNet_B4_448"
args.results_dir = "resultados"
args.seed = 0

# Training
args.weight_decay = 0.05
args.lr = 5e-4
args.opt = 'adamw'
args.momentum = 0.9
args.epochs = 15
args.batch_size = 64
args.clip_grad = None

args.alpha = 0.5
args.tau = 1

# Modelo
args.model_dir = "/datasets/melanoma-winning-models/melanoma-winning-models"
args.kernel_type = "9c_b4ns_448_ext_15ep-newfold"
args.enet_type = "tf_efficientnet_b4_ns"
args.out_dim = 9
args.use_meta = False
args.image_size = 448
args.n_meta_dim = "512,128"
args.eval = "best"
args.fold = 0

# Dataset
args.data_dir = "./data"
args.data_folder = 512

wandb.init(project="Melanoma Distil", entity="mfidabel", config=vars(args))

criterion = nn.CrossEntropyLoss()
sigmoid = nn.Sigmoid()
device = torch.device('cuda')
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# %% [markdown]
# # Modelos

# %% [markdown]
# - EfficientNet
# - Resnet
# - Seresnext

# %%
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)


class Effnet_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        super(Effnet_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.classifier.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out


class Resnest_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        super(Resnest_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        self.enet = resnest101(pretrained=pretrained)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.fc.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.fc = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out


class Seresnext_Melanoma(nn.Module):
    def __init__(self, enet_type, out_dim, n_meta_features=0, n_meta_dim=[512, 128], pretrained=False):
        super(Seresnext_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        if pretrained:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        else:
            self.enet = se_resnext101_32x4d(num_classes=1000, pretrained=None)
        self.enet.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.enet.last_linear.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.last_linear = nn.Identity()

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out

# %% [markdown]
# - DeiT

# %%
def crear_modelo_deit(img_size: int, num_classes: int, pretrained=True, model_name = "deit_base_distilled_patch16_384"):
    model_deit = timm.create_model(model_name, 
                          img_size=(img_size, img_size), 
                          pretrained=pretrained,
                          num_classes=num_classes)

    model_deit.set_distilled_training()

    model_deit = model_deit.to(device)
    
    return model_deit

# %% [markdown]
# # Dataset

# %%
class MelanomaDataset(Dataset):
    def __init__(self, csv, mode, meta_features, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            res = self.transform(image=image)
            image = res['image'].astype(np.float32)
        else:
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        if self.use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][self.meta_features]).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            return data
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long()

# %%
def get_transforms(image_size):

    transforms_train = albumentations.Compose([
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(limit=0.2, p=0.75),
        albumentations.RandomContrast(limit=0.2, p=0.75),
        albumentations.OneOf([
            albumentations.MotionBlur(blur_limit=5),
            albumentations.MedianBlur(blur_limit=5),
            albumentations.GaussianBlur(blur_limit=5),
            albumentations.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            albumentations.OpticalDistortion(distort_limit=1.0),
            albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        albumentations.CLAHE(clip_limit=4.0, p=0.7),
        albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val

# %%
def get_meta_data(df_train, df_test):

    # One-hot encoding of anatom_site_general_challenge feature
    concat = pd.concat([df_train['anatom_site_general_challenge'], df_test['anatom_site_general_challenge']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    df_train = pd.concat([df_train, dummies.iloc[:df_train.shape[0]]], axis=1)
    df_test = pd.concat([df_test, dummies.iloc[df_train.shape[0]:].reset_index(drop=True)], axis=1)
    # Sex features
    df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
    df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
    df_train['sex'] = df_train['sex'].fillna(-1)
    df_test['sex'] = df_test['sex'].fillna(-1)
    # Age features
    df_train['age_approx'] /= 90
    df_test['age_approx'] /= 90
    df_train['age_approx'] = df_train['age_approx'].fillna(0)
    df_test['age_approx'] = df_test['age_approx'].fillna(0)
    df_train['patient_id'] = df_train['patient_id'].fillna(0)
    # n_image per user
    df_train['n_images'] = df_train.patient_id.map(df_train.groupby(['patient_id']).image_name.count())
    df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image_name.count())
    df_train.loc[df_train['patient_id'] == -1, 'n_images'] = 1
    df_train['n_images'] = np.log1p(df_train['n_images'].values)
    df_test['n_images'] = np.log1p(df_test['n_images'].values)
    # image size
    train_images = df_train['filepath'].values
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(tqdm(train_images)):
        train_sizes[i] = os.path.getsize(img_path)
    df_train['image_size'] = np.log(train_sizes)
    test_images = df_test['filepath'].values
    test_sizes = np.zeros(test_images.shape[0])
    for i, img_path in enumerate(tqdm(test_images)):
        test_sizes[i] = os.path.getsize(img_path)
    df_test['image_size'] = np.log(test_sizes)

    meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')]
    n_meta_features = len(meta_features)

    return df_train, df_test, meta_features, n_meta_features

# %%
def get_df(kernel_type, out_dim, data_dir, data_folder, use_meta):

    # 2020 data
    df_train = pd.read_csv(os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}', 'train.csv'))
    df_train = df_train[df_train['tfrecord'] != -1].reset_index(drop=True)
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}/train', f'{x}.jpg'))

    if 'newfold' in kernel_type:
        tfrecord2fold = {
            8:0, 5:0, 11:0,
            7:1, 0:1, 6:1,
            10:2, 12:2, 13:2,
            9:3, 1:3, 3:3,
            14:4, 2:4, 4:4,
        }
    elif 'oldfold' in kernel_type:
        tfrecord2fold = {i: i % 5 for i in range(15)}
    else:
        tfrecord2fold = {
            2:0, 4:0, 5:0,
            1:1, 10:1, 13:1,
            0:2, 9:2, 12:2,
            3:3, 8:3, 11:3,
            6:4, 7:4, 14:4,
        }
    df_train['fold'] = df_train['tfrecord'].map(tfrecord2fold)
    df_train['is_ext'] = 0

    # 2018, 2019 data (external data)
    df_train2 = pd.read_csv(os.path.join(data_dir, f'jpeg-isic2019-{data_folder}x{data_folder}', 'train.csv'))
    df_train2 = df_train2[df_train2['tfrecord'] >= 0].reset_index(drop=True)
    df_train2['filepath'] = df_train2['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-isic2019-{data_folder}x{data_folder}/train', f'{x}.jpg'))
    if 'newfold' in kernel_type:
        df_train2['tfrecord'] = df_train2['tfrecord'] % 15
        df_train2['fold'] = df_train2['tfrecord'].map(tfrecord2fold)
    else:
        df_train2['fold'] = df_train2['tfrecord'] % 5
    df_train2['is_ext'] = 1

    # Preprocess Target
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))

    if out_dim == 9:
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
    elif out_dim == 4:
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('DF', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('AK', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('SCC', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('VASC', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('BCC', 'unknown'))
    else:
        raise NotImplementedError()

    # concat train data
    df_train = pd.concat([df_train, df_train2]).reset_index(drop=True)

    # test data
    df_test = pd.read_csv(os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}', 'test.csv'))
    df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}/test', f'{x}.jpg'))

    if use_meta:
        df_train, df_test, meta_features, n_meta_features = get_meta_data(df_train, df_test)
    else:
        meta_features = None
        n_meta_features = 0

    # class mapping
    diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train.diagnosis.unique()))}
    df_train['target'] = df_train['diagnosis'].map(diagnosis2idx)
    mel_idx = diagnosis2idx['melanoma']

    return df_train, df_test, meta_features, n_meta_features, mel_idx

# %% [markdown]
# # Destilación

# %% [markdown]
# Ciclo de destilación (Entrenamiento)

# %%
def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                      data_loader: Iterable, optimizer: torch.optim.Optimizer,
                      device: torch.device, epoch: int, loss_scaler, max_norm: float = None,
                      set_training_mode=True, args = None):

    # Pone en entrenamiento
    model.train(set_training_mode)
    metric_logger = MetricLogger(delimiter=" ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt='{value:.6f}'))
    loss_ce = nn.CrossEntropyLoss()
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        # Mandamos al dispositivo (GPU)
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)
            ensamble_output = (outputs[0] + outputs[1])/ 2
            avg_ce_loss_clf = loss_ce(outputs[0], targets)
            avg_ce_loss_dist = loss_ce(outputs[1], targets)
            avg_ce_loss = loss_ce(ensamble_output, targets)
            acc1, acc5 = accuracy(ensamble_output, targets, topk=(1, 5))
            
        loss_value = loss.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
            
        optimizer.zero_grad()
        
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        
        batch_size = samples.shape[0]
        
        metric_logger.update(loss_distill = loss_value,
                             lr = optimizer.param_groups[0]["lr"],
                             loss_ce = avg_ce_loss.item(),
                             loss_ce_clf = avg_ce_loss_clf.item(),
                             loss_ce_dist = avg_ce_loss_dist.item())

        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
       
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    wandb.log({f"train/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}, step=epoch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

# %% [markdown]
# # Evaluación

# %%
@torch.no_grad()
def evaluate(data_loader, model, device, mel_idx, epoch, is_ext=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    LOGITS = []
    PROBS = []
    TARGETS = []

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        LOGITS.append(output.detach().cpu())
        PROBS.append(output.softmax(1).detach().cpu())
        TARGETS.append(target.detach().cpu())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    LOGITS = torch.cat(LOGITS).numpy()
    PROBS = torch.cat(PROBS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()

    acc = (PROBS.argmax(1) == TARGETS).mean() * 100.
    auc = roc_auc_score((TARGETS == mel_idx).astype(float), PROBS[:, mel_idx])
    auc_20 = roc_auc_score((TARGETS[is_ext == 0] == mel_idx).astype(float), PROBS[is_ext == 0, mel_idx])

    wandb.log({f"val/{k}": meter.global_avg for k, meter in metric_logger.meters.items()}, step=epoch)
    wandb.log({
        "val/acc": acc,
        "val/auc": auc,
        "val/auc_20": auc_20
    }, step=epoch)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, acc, auc, auc_20

# %% [markdown]
# # Entrenamiento

# %% [markdown]
# Obtener dataset

# %%
df, df_test, meta_features, n_meta_features, mel_idx = get_df(
    args.kernel_type,
    args.out_dim,
    args.data_dir,
    args.data_folder,
    args.use_meta
)

transforms_train, transforms_val = get_transforms(args.image_size)

# %% [markdown]
# Obtener el modelo teacher

# %%
if args.enet_type == 'resnest101':
    ModelClass = Resnest_Melanoma
elif args.enet_type == 'seresnext101':
    ModelClass = Seresnext_Melanoma
elif 'efficientnet' in args.enet_type:
    ModelClass = Effnet_Melanoma
else:
    raise NotImplementedError()


# %%
fold = args.fold
df_train = df[df['fold'] != fold]
df_valid = df[df['fold'] == fold]
dataset_train = MelanomaDataset(df_train, 'train', meta_features, transform=transforms_train)
dataset_valid = MelanomaDataset(df_valid, 'valid', meta_features, transform=transforms_val)
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, sampler=RandomSampler(dataset_train), num_workers=8)
valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size, num_workers=8)

# %%
model_file = os.path.join(args.model_dir, f'{args.kernel_type}_best_fold{fold}.pth')
model_teacher = ModelClass(
            args.enet_type,
            n_meta_features=n_meta_features,
            n_meta_dim=[int(nd) for nd in args.n_meta_dim.split(',')],
            out_dim=args.out_dim
        )
model_teacher = model_teacher.to(device)

try:  # single GPU model_file
    model_teacher.load_state_dict(torch.load(model_file), strict=True)
except:  # multi GPU model_file
    state_dict = torch.load(model_file)
    state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
    model_teacher.load_state_dict(state_dict, strict=True)

_ = model_teacher.eval()

# %%
criterion = torch.nn.CrossEntropyLoss()
loss_fn = DistillationLoss(base_criterion=criterion,
                           teacher_model=model_teacher,
                           distillation_type="hard",
                           alpha = args.alpha,
                           tau = args.tau
                          )

# %%
model_deit = crear_modelo_deit(args.image_size, args.out_dim)

optimizer = create_optimizer(args, model_deit)
loss_scaler = NativeScaler()

# %%
model_file_output_best  = os.path.join(args.results_dir, "modelos", f'{args.kernel_type}_best_fold{args.fold}.pth')
model_file_output_best_20 = os.path.join(args.results_dir, "modelos", f'{args.kernel_type}_best_20_fold{args.fold}.pth')
model_file_output_final = os.path.join(args.results_dir, "modelos",f'{args.kernel_type}_final_fold{args.fold}.pth')

auc_max = 0.
auc_20_max = 0.

for epoch in range(args.epochs):
    train_stats = train_one_epoch(model = model_deit, criterion=loss_fn ,
                      data_loader = train_loader, optimizer = optimizer,
                      device = device, epoch = epoch + 1, loss_scaler = loss_scaler, 
                      set_training_mode=True, max_norm=args.clip_grad, args = args)
    print(train_stats)
    validation_stats, acc, auc, auc_20 = evaluate(model = model_deit, data_loader = valid_loader, device=device,
                        mel_idx= mel_idx, is_ext=df_valid["is_ext"].values, epoch= epoch + 1)
    print(validation_stats)

    # Guardar modelo

    if auc > auc_max:
        print('auc_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_max, auc))
        torch.save(model_deit.state_dict(), model_file_output_best)
        auc_max = auc

    if auc_20 > auc_20_max:
        print('auc_20_max ({:.6f} --> {:.6f}). Saving model ...'.format(auc_20_max, auc_20))
        torch.save(model_deit.state_dict(), model_file_output_best_20)
        auc_20_max = auc_20

# Detalles finales
torch.save(model_deit.state_dict(), model_file_output_final)
wandb.finish(exit_code=0)


