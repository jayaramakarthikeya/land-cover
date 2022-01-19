
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from utils import mIOU , check_accuracy , save_checkpoint , load_checkpoint , train_fn
from dataset import get_dataloaders
from model import UNET


device = ('cuda' if torch.cuda.is_available() else 'cpu')

#set hyperparams
epochs = 5
lr = 3e-4
model = UNET(in_channels=3,out_channels=7).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
loss_fn = nn.CrossEntropyLoss()


def train():
    best_iou_score = 0.0

    train_loader, val_loader = get_dataloaders()

    train_log_list , val_log_list = [] , []


    for epoch in range(epochs):
        train_log , val_log = {} , {}
        loss = train_fn(train_loader,model,optimizer,loss_fn,device)
        train_log['loss'] = loss
        train_log_list.append(train_log)
        

        # check accuracy
        dice_score , iou = check_accuracy(val_loader, model, device=device)
        val_log['dice_score'] = dice_score 
        val_log['IoU'] = iou
        val_log_list.append(val_log)
        
        if best_iou_score < val_log['IoU']:
            best_iou_score < val_log['IoU']
            # save model
            checkpoint = {
                'epoch':epoch,
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
                "loss":loss,
                "dice_score":dice_score,
                "miou":iou
            }
            save_checkpoint(checkpoint)
            print('Model saved!')


if __name__ == "__main__":
    train()