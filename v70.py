#!/usr/bin/env python
# coding: utf-8

# In[1]:


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
import warnings
warnings.filterwarnings('ignore')

import command
import os
import random
import copy
import math
import shutil
import wandb
import cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from glob import glob

import segmentation_models_pytorch as smp
import timm

from sklearn.model_selection import *
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

from transformers import get_cosine_schedule_with_warmup


# In[2]:


#caformer_m36.sail_in22k_ft_in1k_384


# In[3]:


class CFG:
    FOLD = 4
    FULLDATA = False
    
    image_size = [480, 480]
    seed = 1123
    
    model_name = 'tf_efficientnetv2_l.in21k_ft_in1k'#"tf_efficientnetv2_xl.in21k_ft_in1k"
    v = '70'
    
    device = torch.device('cuda')
    
    n_folds = 5
    n_epochs = 10
    
    lr = 2e-4
    
    train_batch_size = 8
    valid_batch_size = 8
    acc_steps = 2
    
    wd = 1e-6
    n_warmup_steps = 0
    upscale_steps = 1.1
    validate_every = 1
    
    epoch = 0
    global_step = 0 
    literal_step = 0
    
    autocast = True
    

    workers = 0
    
if CFG.FULLDATA:
    CFG.seed = CFG.FOLD

'''
done_version = []
versions = glob("/mnt/md0/wns_triangle/AAA_SEG/TRY1_SEG/*")
for version in versions:
    a = version.split("/")[-1].split("_")[-1].split("v")[-1]
    try:
        a = int(a)
        done_version.append(a)
    except:
        pass
try:
    last_version = max(done_version)
    CFG.v = last_version + 1
except:
    CFG.v = 0
'''

OUTPUT_FOLDER = f"/mnt/md0/wns_triangle/AAA_SEG/TRY1_SEG/{CFG.model_name}_v{CFG.v}"
CFG.cache_dir = OUTPUT_FOLDER + '/cache/'
os.makedirs(CFG.cache_dir, exist_ok=1)

seed_everything(CFG.seed)
CFG.v


# In[ ]:





# In[4]:


data = pd.read_csv("/mnt/md0/wns_triangle/train/train.csv")
data["label"].unique()


# In[ ]:





# In[5]:


class WNSDataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self. transforms = transforms
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        row = self.data.iloc[i]
        file = row["filename"].replace(".jpg", ".npy")
        label = row["label"]
        ids = row["image_id"]
        
        image = np.load(f"/mnt/md0/wns_triangle/train/images/{file}")
        
        if self.transforms:
            transformed = self.transforms(image=image)
            
            image = transformed["image"]
            if image.dtype==torch.uint8: image = image.float() / 255
        
        label = torch.as_tensor([label])
        
        return {"images": image,
               "labels": label,
               "ids": ids}
            


# In[6]:


folds = [*StratifiedKFold(n_splits=CFG.n_folds).split(data, data["label"])]

def get_loaders():
    train_data = data.iloc[folds[CFG.FOLD][0]]
    valid_data = data.iloc[folds[CFG.FOLD][1]]
    
    train_augs = A.Compose([
        A.Resize(CFG.image_size[0], CFG.image_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #A.RandomBrightnessContrast (brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        #A.RandomRotate90(p=0.5),
        #A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, always_apply=False, p=0.5),
        #A.CoarseDropout(max_holes=8, max_height=8, max_width=8, fill_value=0, always_apply=False, p=0.3),
        #A.Normalize(max_pixel_value=255),
        ToTensorV2(),
    ]) 
    
    valid_augs = A.Compose([
        A.Resize(CFG.image_size[0], CFG.image_size[1]),
        #A.Normalize(max_pixel_value=255),
        ToTensorV2()
    ])
    
    train_dataset = WNSDataset(train_data, train_augs)
    valid_dataset = WNSDataset(valid_data, valid_augs)
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, shuffle=True, num_workers=CFG.workers, pin_memory=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_batch_size, shuffle=False, num_workers=CFG.workers, pin_memory=False)
    
    CFG.steps_per_epoch = math.ceil(len(train_loader) / CFG.acc_steps)
    
    return train_loader, valid_loader

train_loader, valid_loader = get_loaders()

for d in tqdm(valid_loader):
     break


# In[7]:


plt.imshow(d['images'][0].permute(1,2,0))


# In[ ]:





# In[8]:


data.label.sum() / len(data)


# In[9]:


timm.models


# In[10]:


class Model_conv(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = timm.create_model(CFG.model_name, pretrained=True, num_classes=0, global_pool='')
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.feats = self.backbone.num_features
        
        self.head = nn.Linear(self.feats, 1)
    
    def forward(self, inp):
        features = self.backbone(inp)
        
        features = self.avgpool(features).flatten(1, 3)
        
        logits = self.head(features)
        
        return logits


# In[11]:


class Model_trans(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = timm.create_model(CFG.model_name, pretrained=True, num_classes=0, global_pool='')
        
        self.feats = self.backbone.num_features
        
        self.head = nn.Linear(self.feats, 1)
    
    def forward(self, inp):
        features = self.backbone(inp)
        
        features = features.mean(1)
        
        logits = self.head(features)
        
        return logits


# In[12]:


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        
    def forward(self, outputs, targets):
        outputs = outputs.float()
        targets = targets.float()
        
        loss1 = self.bce(outputs, targets)
        
        return loss1


# In[13]:


def get_optmizer_scheduler_criterion_scaler(model):
    criterion = CustomLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=CFG.n_warmup_steps, num_training_steps=CFG.steps_per_epoch*CFG.n_epochs*CFG.upscale_steps)
        
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.autocast)
    
    return optimizer, scheduler, criterion, scaler


# In[14]:


def train_one_epoch(model, loader):
    model.train()
    running_loss = 0.0

    if is_main_process(): bar = tqdm(loader, bar_format='{n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}')
    else: bar = loader
    
    for step, data in enumerate(bar):
        step += 1
        
        images = data['images'].cuda()
        targets = data['labels'].cuda()
        
        with torch.cuda.amp.autocast(enabled=CFG.autocast):
            outputs = model(images)
            
        #print(targets.shape, logits.shape)
        #print(targets.dtype, logits.dtype)
        
        loss = criterion(outputs, targets)
        
        running_loss += (loss - running_loss) * (1 / step)
        
        loss = loss / CFG.acc_steps
        scaler.scale(loss).backward()
        
        if step % CFG.acc_steps == 0 or step == len(bar):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            CFG.global_step += 1
        
        CFG.literal_step += 1
        
        #lr = "{:2e}".format(next(optimizer.param_groups)['lr'])
        lr = "{:2e}".format(optimizer.param_groups[0]['lr'])
        
        if is_main_process():
            bar.set_postfix(loss=running_loss.item(), lr=float(lr), step=CFG.global_step)
        
        #break
        
        #if step==10: break
        
        #dist.barrier()
    
    if is_main_process():
        #torch.save(model.module.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}-{CFG.epoch}.pth")
        torch.save(model.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth")
        
        
def valid_one_epoch(path, loader, running_dist=False, debug=False):
    #model = Model()
    st = torch.load(path, map_location=f"cpu")
    model.eval()
    model.cuda()
    model.load_state_dict(st, strict=False)
    
    if is_main_process(): bar = tqdm(loader, bar_format='{n_fmt}/{total_fmt} {elapsed}<{remaining} {postfix}')
    else: bar = loader
    
    running_loss = 0.
    
    OUTPUTS = []
    TARGETS = []
    MASKS_TARGETS = []
    IDS = []
    
    for step, data in enumerate(bar):
        with torch.no_grad():
            images = data['images'].cuda()
            targets = data['labels'].cuda()
            ids = data['ids']
            
            with torch.cuda.amp.autocast(enabled=CFG.autocast):
                logits = model(images)
            
            #logits = logits[:, :, :9]
            #targets = targets[:, :, :9]
            
            outputs = logits.float().sigmoid().detach().cpu()#.numpy()
            targets = targets.float().detach().cpu()#.numpy()
            
            #ids = np.array(ids)
            
            OUTPUTS.extend(outputs)
            TARGETS.extend(targets)
            IDS.extend(ids)
    
    OUTPUTS = np.stack(OUTPUTS)
    TARGETS = np.stack(TARGETS)
    IDS = np.stack(IDS)
    
    threshold = 0.5
    score = f1_score(TARGETS, OUTPUTS>threshold, average='macro')
    
    wandb.log({"F1": score})
    print(f"EPOCH {CFG.epoch+1} | F1 Score {score}")
    #print(class_metric)
    
    if debug:
        return score, None, None, None
    
    return score

def run(model, get_loaders):
    if is_main_process():
        epochs = []
        scores = []
    
    best_score = float('-inf')
    for epoch in range(CFG.n_epochs):
        CFG.epoch = epoch
        
        train_loader, valid_loader = get_loaders()
        
        train_one_epoch(model, train_loader)
        
        #dist.barrier()
        
        if is_main_process():
            if (CFG.epoch+1)%CFG.validate_every==0 or epoch==0:
                score, OUTPUTS, TARGETS, IDS = valid_one_epoch(f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth", valid_loader, debug=True, running_dist=False)
        
        #dist.barrier()
        
        if is_main_process():
            epochs.append(epoch)
            scores.append(score)
            wandb.config.update({"Latest_Score": score}, allow_val_change=True)
            if score > best_score:
                print("SAVING BEST!")
                torch.save(model.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}_best.pth")
                best_score = score
            
                command.run(['rm', '-r', CFG.cache_dir])
                pass
            
            wandb.config.update({"Best_F1": score}, allow_val_change=True)
            
            os.makedirs(CFG.cache_dir, exist_ok=1)
    return best_score


# In[15]:


#OUTPUTS, TARGETS = valid_one_epoch(f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth", valid_loader, debug=True, running_dist=False)


# In[16]:


#from sklearn.metrics import *
#balanced_accuracy_score(TARGETS, OUTPUTS)


# In[17]:


#OUTPUTS[TARGETS==0]


# In[18]:


f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth"


# In[19]:


name = f"Version-{CFG.v}-model-{CFG.model_name}"
logs_directory = f"./WANDB/Version-{CFG.v}-model-{CFG.model_name}/"
os.makedirs(logs_directory, exist_ok=True)

wandb.init(project="WNS_Triangle", name=name, dir = logs_directory)
wandb.config.update({"lr": CFG.lr, "model": CFG.model_name, "img size": CFG.image_size, "Version": CFG.v, "out folder": OUTPUT_FOLDER.split("/")[-1], "Epochs": CFG.n_epochs}, allow_val_change=True)
wandb.log({"output folder": OUTPUT_FOLDER})
def is_main_process():
    return 1

train_loader, valid_loader = get_loaders()
########################################

model = Model_conv()

########################################
model.cuda()

if is_main_process():
    torch.save(model.state_dict(), f"{OUTPUT_FOLDER}/{CFG.FOLD}.pth")
    
    
optimizer, scheduler, criterion, scaler = get_optmizer_scheduler_criterion_scaler(model)
    
best_dice = run(model, get_loaders)
wandb.log({"Best Dice": best_dice, "Version": CFG.v})
shutil.copy("./train1.ipynb", f"./WANDB/Version-{CFG.v}-model-{CFG.model_name}/V{CFG.v}-NB.ipynb")
#!jupyter nbconvert --to script ./WANDB/Version-{CFG.v}-model-{CFG.model_name}/V{CFG.v}-NB.ipynb
wandb.finish()


# In[ ]:





# In[ ]:


import sys
sys.exit(0)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




