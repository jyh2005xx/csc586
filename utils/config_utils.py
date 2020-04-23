# Partly from https://raw.githubusercontent.com/vcg-uvic/lf-net-release/master/common/argparse_utils.py

import argparse
import sys

def str2bool(v):
    return v.lower() in ('true', '1', 'yes', 'y', 't')

def get_oicr_config():
    parser = argparse.ArgumentParser()
    ## --- General Settings
    # Use Seed or Not
    parser.add_argument('--set_seed', type=str2bool, default=True)
    # Run Name
    parser.add_argument('--name', type=str, default='test')
    # Dataset base path
    parser.add_argument('--dataset_dir', type=str, default='/home/yuhe/dataset')
    # Model base path
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--result_dir', type=str, default='./results')
    # Choose dataset
    parser.add_argument('--dataset', type=str, default='VOC2007')
    # Number of Epoch
    parser.add_argument('--epochs', type=int, default=300)
    # Save Weight
    parser.add_argument('--save_weights',type=str2bool, default=False)
    # Resume
    parser.add_argument('--resume',type=str2bool, default=True)
    # Validation
    parser.add_argument('--valid_period', type=int, default=10000)
    parser.add_argument('--valid_flag', type=str2bool, default=True)
    # Tensorboard 
    parser.add_argument('--eval_period', type=int, default=400)

    parser.add_argument('--log_dir', type=str, default='logs/')

    parser.add_argument('--lr', type=float, default=1e-6)

    parser.add_argument('--tune_vgg',type=str2bool,default=False)

    # wsdd or ocir
    parser.add_argument('--net',type=str,default='oicr')
    ## --- Classifier Settings 
    # Where to Extract Patches 
    # parser.add_argument('--patch_from_images', type=str2bool, default=False)
    # Patch Size
    parser.add_argument('--patch_size', type=int, default=7)    

    ## --- Experiment Comment
    parser.add_argument('--comment', type=str, default='')
    

    config = parser.parse_args()

    return config

def print_config(config):
    print('---------------------- CONFIG ----------------------')
    print()
    args = list(vars(config))
    args.sort()
    for arg in args:
        print(arg.rjust(25,' ') + '  ' + str(getattr(config, arg)))
    print()
    print('----------------------------------------------------')

def config_to_string(config):
    string = '\n\n'
    string += 'python ' + ' '.join(sys.argv)
    string += '\n\n'
    # string += '---------------------- CONFIG ----------------------\n'
    args = list(vars(config))
    args.sort()
    for arg in args:
        string += arg.rjust(25,' ') + '  ' + str(getattr(config, arg)) + '\n\n'
    # string += '----------------------------------------------------\n'
    return string
