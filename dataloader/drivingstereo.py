import torch.utils.data as data
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, mode='train'):
    all_left_img=[]
    all_right_img=[]
        
    if mode == 'train':
        driving_stereo_dir = filepath + '/train/'
        imm_l = os.listdir(driving_stereo_dir+'/train-left-image/')
        for im in imm_l:
            if is_image_file(driving_stereo_dir+'/train-left-image/'+im):
                all_left_img.append(driving_stereo_dir+'/train-left-image/'+im)

        imm_r = os.listdir(driving_stereo_dir+'/train-right-image/')
        for im in imm_r:
            if is_image_file(driving_stereo_dir+'/train-right-image/'+im):
                all_right_img.append(driving_stereo_dir+'/train-right-image/'+im)
        return all_left_img, all_right_img
    else:
        driving_stereo_dir = filepath + '/test/'
        imm_l = os.listdir(driving_stereo_dir+'/left-image-full-size/')
        for im in imm_l:
            if is_image_file(driving_stereo_dir+'/left-image-full-size/'+im):
                all_left_img.append(driving_stereo_dir+'/left-image-full-size/'+im)

        imm_r = os.listdir(driving_stereo_dir+'/right-image-full-size/')
        for im in imm_r:
            if is_image_file(driving_stereo_dir+'/right-image-full-size/'+im):
                all_right_img.append(driving_stereo_dir+'/right-image-full-size/'+im)

        return all_left_img, all_right_img
    


