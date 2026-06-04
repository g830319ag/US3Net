import torch.utils.data as data
from PIL import Image
import os
import os.path
from glob import glob

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, split='training'):
    all_left_img=[]
    all_right_img=[]
        
    left_img_list = sorted(glob(os.path.join(filepath, f'two_view_{split}/*/im0.png')) )
    right_img_list = sorted(glob(os.path.join(filepath, f'two_view_{split}/*/im1.png')) )

    for img1, img2 in zip(left_img_list, right_img_list):
        if is_image_file(img1):
            all_left_img.append(img1)
        if is_image_file(img2):
            all_right_img.append(img2)
        return all_left_img, all_right_img
    


