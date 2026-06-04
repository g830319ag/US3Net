from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

from models import *
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

parser = argparse.ArgumentParser(description='US3Net')
parser.add_argument('--model', default='US3Net',
                        choices=['RealTimeStereo',
                                 'US3Net'],
                        help='select model (default: RealTimeStereo)')
parser.add_argument('--maxdisp', type=int ,             default=192, help='maxium disparity')
parser.add_argument('--data_path',                      default='./test_img/kitti2015', help='datapath')
parser.add_argument('--load_cpt_path',                  default='./US3Net_kitti_training.cpt', help='load model')
parser.add_argument('--no-cuda', action='store_true',   default=False, help='enables CUDA training')
parser.add_argument('--min_depth', type=float,          default=1e-8, help='minimum depth for evaluation')
parser.add_argument('--max_depth', type=float,          default=80, help='maximum depth for evaluation')
args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()

print('init model...')
if args.model == 'RealTimeStereo':
    from models import RTStereoNet as stereoNet
elif args.model == 'US3Net':
    from models import US3Net as stereoNet
else:
    print('no model')

model = stereoNet(args.maxdisp)
if args.cuda:
    model.cuda()

if args.load_cpt_path is not None:
    print(f'loading model from {args.load_cpt_path} ...')
    state_dict = torch.load(args.load_cpt_path)
    from collections import OrderedDict
    model_state_dict = OrderedDict()

    for k, v in state_dict['state_dict'].items():
        k = k.replace('module.', '')
        model_state_dict[k] = v

    
    model.load_state_dict(model_state_dict)
model.eval()
    
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def read_gt_disp_kitti(gt_disp_path):
    gt_disp = Image.open(gt_disp_path)
    gt_disp = np.array(gt_disp, dtype=np.float32) / 256.
    return gt_disp

def read_image_pair(imgL_path, imgR_path):
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(**normal_mean_var)])    

    imgL = Image.open(imgL_path).convert('RGB')
    imgR = Image.open(imgR_path).convert('RGB')
    imgL = infer_transform(imgL)
    imgR = infer_transform(imgR) 
    
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1]//16       
        top_pad = (times+1)*16 -imgL.shape[1]
    else:
        top_pad = 0
    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2]//16                       
        right_pad = (times+1)*16-imgL.shape[2]
    else:
        right_pad = 0    

    imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
    imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)
    return imgL, imgR, top_pad, right_pad

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

width_to_focal = dict()
width_to_focal[1242] = 721.5377
width_to_focal[1241] = 718.856
width_to_focal[1224] = 707.0493
width_to_focal[1238] = 718.3351 
width_to_focal[1232] = 718.3351  
width_to_focal[1226] = 707.0912

def convert_disps_to_depths_kitti(gt_disparity, pred_disparity):

    gt_disp = gt_disparity
    _, width = gt_disp.shape
    pred_disp = pred_disparity
    mask = gt_disp > 0

    gt_depth = width_to_focal[width] * 0.54 / (gt_disp + (1.0 - mask))
    pred_depth = width_to_focal[width] * 0.54 / pred_disp
    
    return gt_depth, pred_depth, pred_disp

def main():
    img_name = os.listdir(args.data_path+'/RGB_left')
    num_samples = len(img_name)
    rms     = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel  = np.zeros(num_samples, np.float32)
    d1_all  = np.zeros(num_samples, np.float32)
    a1      = np.zeros(num_samples, np.float32)
    a2      = np.zeros(num_samples, np.float32)
    a3      = np.zeros(num_samples, np.float32)

    img_name = os.listdir(args.data_path + '/RGB_left')
    for i, _name in tqdm(enumerate(img_name)):
        imgL_path = args.data_path + '/RGB_left/' + _name
        imgR_path = args.data_path + '/RGB_right/' + _name
        gt_disp_path = args.data_path + '/disp_occ_0/' + _name
        imgL, imgR, top_pad, right_pad = read_image_pair(imgL_path, imgR_path)
        gt_disp = read_gt_disp_kitti(gt_disp_path)
        if args.cuda:
            imgL, imgR = imgL.cuda(), imgR.cuda()
            
        with torch.no_grad():
            pred_disp = model(imgL,imgR)[0]
        pred_disp = torch.squeeze(pred_disp)
        pred_disp = pred_disp.data.cpu().numpy()
        pred_disp = (pred_disp*256).astype('uint16')
        pred_disp = np.array(pred_disp, dtype=np.float32) / 256.
        if top_pad !=0 and right_pad != 0:
                pred_disp = pred_disp[top_pad:,:-right_pad]
        elif top_pad ==0 and right_pad != 0:
            pred_disp = pred_disp[:,:-right_pad]
        elif top_pad !=0 and right_pad == 0:
            pred_disp = pred_disp[top_pad:,:]
        else:
            pred_disp = pred_disp
        
        gt_depth, pred_depth, pred_disparitie = convert_disps_to_depths_kitti(gt_disp, pred_disp)
        pred_depth[pred_depth < args.min_depth] = args.min_depth
        pred_depth[pred_depth > args.max_depth] = args.max_depth
        
        # calculate d1_all
        mask = gt_disp > 0
        disp_diff = np.abs(gt_disp[mask] - pred_disparitie[mask])
        bad_pixels = np.logical_and(disp_diff >= 3, (disp_diff / gt_disp[mask]) >= 0.05)
        d1_all[i] = 100.0 * bad_pixels.sum() / mask.sum()
        abs_rel[i], sq_rel[i], rms[i], log_rms[i], a1[i], a2[i], a3[i] = compute_errors(gt_depth[mask], pred_depth[mask])

    print("---------Result---------")
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('abs_rel', 'sq_rel', 'rms', 'log_rms', 'd1_all', 'a1', 'a2', 'a3'))
    print("{:10.4f}, {:10.4f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}, {:10.3f}".format(abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), d1_all.mean(), a1.mean(), a2.mean(), a3.mean()))

if __name__ == '__main__':
   main()