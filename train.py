import os
import argparse
from models import *
from tqdm import tqdm
from trainer import Trainer
import torch
from torch.utils.data import DataLoader

def load_args():
    parser = argparse.ArgumentParser(description='ES3Net')
    parser.add_argument('--dataset', default='self',
                        choices=['kitti_raw', 'kitti_eigen', 'self', 'sceneflow', 'eth3d', 'drivingstereo'],
                        help='dataset for choose loader')
    
    parser.add_argument('--eigen_path', default=None,
                        help='path to eigen dataset text file')
    parser.add_argument('--data_path', default=None,
                        help='path to data')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--model', default='RealTimeStereo_depthwise',
                        choices=['RealTimeStereo_depthwise_post2d',
                                 'RealTimeStereo_shuffle',
                                 'RealTimeStereo_depthwise',
                                 'RealTimeStereo_coordattn',
                                 'RealTimeStereo_pixelshuffle',
                                 'RealTimeStereo', 
                                'stackhourglass', 
                                'stackhourglass2d', 
                                'basic'],
                        help='select model (default: RealTimeStereo)')
    parser.add_argument('--load_cpt_path', default=None,
                        help='path for loading model checkpoint')    
    parser.add_argument('--save_cpt_dir', type=str, default='./cpts',
                        help='directory for saving model checkpoint')

    parser.add_argument('--save_freq', type=int, default=1,
                        help='frequency of saving model checkpoint')

    parser.add_argument('--maxdisp', type=int, default=192,
                        help='maxium disparity')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disable CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dual_transform', default='flip',
                        choices=['flip', 'rotate'],
                        help='dula transform mode (default: flip)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training from checkpoint')

    # training hyper-parameters
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 for Adam optimizer (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 for Adam optimizer (default: 0.999)')

    parser.add_argument('--lr_loss_weight', type=float, default=0.01,
                        help='weight for lr loss (default: 0.01)')
    parser.add_argument('--smth_loss_weight', type=float, default=10,
                        help='weight for smooth loss (default: 10)')

    parser.add_argument('--train_batch_size', type=int, default=128,
                        help='batch size for training (default: 12)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of workers for training (default: 8)')

    parser.add_argument('--multi_scale', action='store_true', default=False,
                        help='use multi-scale reconstruction for training (default: False)')
    parser.add_argument('--only_recon_epoch', type=int, default=15,
                        help='only train with reconstruction loss for the first N epochs (default: 15)')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = load_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True        

    print('init dataloader...')
    if args.dataset == 'kitti_eigen':
        from dataloader import KITTIloaderEigen as lt
        from dataloader import allloader as DA
        all_left_img, all_right_img = lt.dataloader(args.eigen_path, args.data_path)
    elif args.dataset == 'sceneflow':
        from dataloader import listflowfile as lt
        from dataloader import SceneFlowLoader as DA
        all_left_img, all_right_img = lt.dataloader(args.data_path)
    elif args.dataset == 'drivingstereo':
        from dataloader import drivingstereo as lt
        from dataloader import allloader as DA
        all_left_img, all_right_img = lt.dataloader(args.data_path)
    else:
        from dataloader import allimage as lt
        from dataloader import allloader as DA
        all_left_img, all_right_img = lt.dataloader(args.data_path)
    # elif args.dataset == 'eth3d':
    #     from dataloader import eth3d as lt
    #     from dataloader import allloader as DA
    #     all_left_img, all_right_img = lt.dataloader(args.data_path, split='training')
    

    TrainLoader = torch.utils.data.DataLoader(
         DA.myImageFloder(all_left_img, all_right_img, True), 
         batch_size= args.train_batch_size, shuffle= True, num_workers= args.num_workers, drop_last=False)

    print('init model...')
    if args.model == 'US3Net':
        from models import US3Net as stereoNet
    elif args.model == 'RealTimeStereo':
        from models import RTStereoNet as stereoNet
    else:
        print('no model')
        
    model = stereoNet(args.maxdisp, args.multi_scale)    
    print('init trainer...')
    trainer = Trainer(model, TrainLoader, args.multi_scale, 
                    args.only_recon_epoch, args.dual_transform)
    model.train()
    print('init optimizer...')
    trainer.init_optimzer(args.lr, (args.beta1, args.beta2))
        
    print('set loss weight...')
    trainer.set_loss_weight(smth_loss_weight= args.smth_loss_weight, 
                            lr_loss_weight= args.lr_loss_weight)
         
    if args.resume:
        print(f'loading model from {args.load_cpt_path} ...')
        state_dict = torch.load(args.load_cpt_path)
        from collections import OrderedDict
        model_state_dict = OrderedDict()
        for k, v in state_dict['state_dict'].items():
            k = k.replace('module.', '')
            model_state_dict[k] = v
        model.load_state_dict(model_state_dict)

        if 'optimizer' in state_dict.keys():
            trainer.optimizer.load_state_dict(state_dict['optimizer'])
        if 'epoch' in state_dict.keys():
            start_epoch = state_dict['epoch']

    else:
        start_epoch = 0
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    
    if args.cuda:
        trainer.cuda()
    
    trainer.fit(start_epoch, args.epochs, args.save_cpt_dir, args.save_freq)
