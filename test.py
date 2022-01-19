
import torch
import numpy as np
from utils import color_code_segmentation , reverse_one_hot , visualize
import random
from dataset import get_test_data , transforms , select_class_rgb_values
from model import UNET

def test():
    model = UNET(in_channels=3,out_channels=7)
    checkpoint = torch.load('../input/checkpoint/my_checkpoint.pth.tar',map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    _ , test_transform = transforms()
    test_dataset = get_test_data(test_transform)

    for i in range(10):
        
        idx = random.randint(60,100)

        image = test_dataset[idx]
        x_tensor = image.unsqueeze(0)
        # Predict test image
        pred_mask = model(x_tensor.float())
        pred_mask = pred_mask.detach().squeeze().numpy()
        
        # Convert pred_mask from `CHW` format to `HWC` format
        pred_mask = np.transpose(pred_mask,(1,2,0))
        
        # Get prediction channel corresponding to foreground
        pred_mask = color_code_segmentation(reverse_one_hot(pred_mask,axis=-1), select_class_rgb_values)
        
        
        
        visualize(
            original_image = image.numpy().transpose(1,2,0),
            predicted_mask = pred_mask
        )
        
        del image , pred_mask

if __name__ == '__main__':
    test()