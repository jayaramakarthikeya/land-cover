import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm



#helper function for data visualization


def visualize(**images):
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx , (name,image) in enumerate(images.items()):
        plt.subplot(1,n_images,idx+1)
        plt.xticks([])
        plt.yticks([])
        plt.title(name.replace('_',' ').title(),fontsize=20)
        plt.imshow(image)
        
    plt.show()
    
    
#perform one hot encoding on the label
def one_hot_encode(label,label_values):
    
    semantic_map = []
    for color in label_values:
        eq = np.equal(label,color)
        class_map = eq.all(axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map,axis=-1)
    
    return semantic_map

#perform reverse one-hot-encoding on labels/preds
def reverse_one_hot(image,axis):
    
    x = np.argmax(image,axis=axis)
    return x



def color_code_segmentation(image,label_values):
    color_codes = label_values
    x = color_codes[image.astype('int')]
    
    return x



class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, inp, target):

        ce_loss = F.cross_entropy(inp, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def mIOU(label, pred, num_classes=7):
    pred = F.softmax(pred, dim=1)              
    pred = torch.argmax(pred, dim=1).squeeze(1)
    iou_list = list()
    present_iou_list = list()

    pred = pred.view(-1)
    label = reverse_one_hot(label.cpu(),axis=1).reshape(-1)
    # Note: Following for loop goes from 0 to (num_classes-1)
    # and ignore_index is num_classes, thus ignore_index is
    # not considered in computation of IoU.
    for sem_class in range(num_classes):
        pred_inds = (pred == sem_class)
        target_inds = (label == sem_class)
        if target_inds.long().sum().item() == 0:
            iou_now = float('nan')
        else: 
            intersection_now = (pred_inds[target_inds]).long().sum().item()
            union_now = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection_now
            iou_now = float(intersection_now) / float(union_now)
            present_iou_list.append(iou_now)
        iou_list.append(iou_now)
    return np.mean(present_iou_list)


def check_accuracy(loader, model, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou_ = []
    iou = 0.0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.permute(0,3,1,2).to(device)
            mask_pred = model(x.float())
            preds = F.softmax(mask_pred)
            preds = (preds > 0.5).float()
            #print(preds.shape,y.shape)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )
            iou_.append(mIOU(y,mask_pred))
            del mask_pred
            torch.cuda.empty_cache()
    
    iou = sum(iou_) / len(iou_)
    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)} , IoU score: {iou} ")
    model.train()
    
    return (dice_score/len(loader)) , iou


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def train_fn(loader,model,optimizer,loss_fn,device):
    
    loop = tqdm(loader)
    
    loss_ = []
    
    for batch_idx , (data,targets) in enumerate(loop):
        
        data = data.to(device)
        targets = targets.permute(0,3,1,2)
        model.train()
        #forward
        
        preds = model(data)
        loss = loss_fn(preds,reverse_one_hot(targets.long(),axis=1).to(device))
            
        #backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        del preds
        
        
        #update tqdm loop
        loop.set_postfix(loss=loss.item())
        
        loss_.append(loss.item())
        torch.cuda.empty_cache()
        
    return sum(loss_) / len(loss_)