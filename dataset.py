import os , cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from utils import visualize , color_code_segmentation , reverse_one_hot , one_hot_encode
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset , DataLoader
from torch.utils.data import random_split

csv_file_path = '../input/deepglobe-land-cover-classification-dataset/metadata.csv'
root_dir = '../input/deepglobe-land-cover-classification-dataset/'

class_dict = pd.read_csv(os.path.join(root_dir,'class_dict.csv'))
class_names = class_dict['name'].tolist()
class_rgb_values = class_dict[['r','g','b']].values.tolist()
select_class_indices = [class_names.index(cls.lower()) for cls in class_names]
select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

class LandCoverDataset(Dataset):
    def __init__(self,is_train_valid,csv_file_path,root_dir,class_rgb_values=None,transform=None):
        self.meta_df = pd.read_csv(csv_file_path)
        self.meta_df = self.meta_df.dropna(subset=['split'])
        self.meta_df = self.meta_df.reset_index(drop=True)
        self.root_dir = root_dir
        self.class_rgb_values = class_rgb_values
        self.is_train_valid = is_train_valid
        if is_train_valid == 'train':
            img_paths = self.meta_df[(self.meta_df.split == 'train')]['sat_image_path']
            mask_paths = self.meta_df[(self.meta_df.split == 'train')]['mask_path']
            
        elif is_train_valid == 'val':
            img_paths = self.meta_df[(self.meta_df.split == 'valid')]['sat_image_path']
            
        else:
            img_paths = self.meta_df[(self.meta_df.split == 'test')]['sat_image_path']
        
        #print(mask_paths)
        self.img_path = [os.path.join(self.root_dir,img_p) for img_p in img_paths]
        if is_train_valid == 'train':
            self.mask_path = [os.path.join(self.root_dir,mask_p) for mask_p in mask_paths]
        self.transform = transform
        
    def __len__(self):
        return len(self.img_path)
    
    def __getitem__(self,idx):
        img_path = self.img_path[idx]
        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        if self.is_train_valid == 'train':
            mask_path = self.mask_path[idx]
            mask = cv2.cvtColor(cv2.imread(mask_path),cv2.COLOR_BGR2RGB)
        
            #one-hot-encode the mask
            mask = one_hot_encode(mask,self.class_rgb_values).astype('float')
        
            if self.transform is not None:
                augmentations = self.transform(image=img,mask=mask)
                img = augmentations['image']
                mask = augmentations['mask']
            
            return img , mask
        
        elif self.is_train_valid == 'test':
            if self.transform is not None:
                img_ten = self.transform(image=img)
                img = img_ten['image']
            return img


def transforms():
    
    train_transform = A.Compose([
        A.RandomCrop(height=1024,width=1024,always_apply=True),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])

    test_transform = A.Compose([
        A.RandomCrop(height=1024,width=1024),
        ToTensorV2(),
    ])

    return train_transform, test_transform


def get_dataloaders(aug_dataset):

    train_transform , _ = transforms()

    aug_dataset = LandCoverDataset('train',csv_file_path,root_dir,class_rgb_values=select_class_rgb_values,transform = train_transform)

    train_size = int(0.9*len(aug_dataset))
    val_size = len(aug_dataset) - train_size

    train_data , val_data = random_split(aug_dataset,[train_size,val_size])

    #loaders
    train_loader = DataLoader(train_data,batch_size=2,shuffle=True,num_workers=2,pin_memory=True)
    val_loader = DataLoader(val_data,batch_size=1,num_workers=2,pin_memory=True)

    return train_loader, val_loader

def get_test_data(test_transform):
    
    csv_file_path = '../input/deepglobe-land-cover-classification-dataset/metadata.csv'
    root_dir = '../input/deepglobe-land-cover-classification-dataset/'

    class_dict = pd.read_csv(os.path.join(root_dir,'class_dict.csv'))
    class_names = class_dict['name'].tolist()
    class_rgb_values = class_dict[['r','g','b']].values.tolist()
    select_class_indices = [class_names.index(cls.lower()) for cls in class_names]
    select_class_rgb_values = np.array(class_rgb_values)[select_class_indices]

    test_dataset = LandCoverDataset('test',csv_file_path,root_dir,class_rgb_values=select_class_rgb_values,transform = test_transform)

    return test_dataset


def main():
    df = pd.read_csv('../input/deepglobe-land-cover-classification-dataset/metadata.csv')
    df.head()

    dataset = LandCoverDataset('train',csv_file_path,root_dir,class_rgb_values=select_class_rgb_values)
    img , mask = dataset[11]

    print(len(dataset))
    visualize(
        original_image = img,
        ground_truth_mask = color_code_segmentation(reverse_one_hot(mask,axis=-1),select_class_rgb_values),
        one_hot_encoded_mask = reverse_one_hot(mask,axis=-1)
        
    )

    train_transform, test_transform = transforms()

    aug_dataset = LandCoverDataset('train',csv_file_path,root_dir,class_rgb_values=select_class_rgb_values,transform = train_transform)
    


    for idx in range(2):
        img , mask = aug_dataset[idx]
        img = img.permute(1,2,0).numpy().astype('float32')
        mask = mask.numpy().astype('float32')
        #print(color_code_segmentation(reverse_one_hot(mask),select_class_rgb_values).shape)
        visualize(
        original_image = img,
        ground_truth_mask = color_code_segmentation(reverse_one_hot(mask,axis=-1),select_class_rgb_values),
        one_hot_encoded_mask = reverse_one_hot(mask,axis=-1)
        
        )


if __name__ == '__main__':
    main()


