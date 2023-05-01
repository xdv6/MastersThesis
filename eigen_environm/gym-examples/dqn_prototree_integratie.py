import sys
import os 
import gym
import gym_game
import pygame

sys.path.append(os.path.abspath('../../Prototree-main'))

from prototree.prototree import ProtoTree
from util.log import Log
from util.args import get_args, save_args, get_optimizer
from util.data import get_dataloaders
from util.init import init_tree
from util.net import get_network, freeze
from util.visualize import gen_vis
from util.analyse import *
from util.save import *
from prototree.train import train_epoch, train_epoch_kontschieder
from prototree.test import eval, eval_fidelity
from prototree.prune import prune
from prototree.project import project, project_with_class_constraints
from prototree.upsample import upsample


import torch
from shutil import copy
from copy import deepcopy
from dqn_util import *


if __name__ == '__main__':
    """
    setup Prototree logging variables
    """

    args = get_args()
    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    # Create a csv log for storing the test accuracy, mean train accuracy and mean loss for each epoch
    log.create_log('log_epoch_overview', 'epoch', 'test_acc', 'mean_train_acc', 'mean_train_crossentropy_loss_during_epoch')
    # Log the run arguments
    save_args(args, log.metadata_dir)
    if not args.disable_cuda and torch.cuda.is_available():
        # device = torch.device('cuda')
        device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
    else:
        device = torch.device('cpu') 

    # Log which device was actually used
    log.log_message('Device used: '+str(device))

    # Create a log for logging the loss values
    log_prefix = 'log_train_epochs'
    log_loss = log_prefix+'_losses'
    log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')


    """
    setup DQN logging variables
    """
    win_count = 0
    achieved_rewards = torch.tensor([], device=device)
    running_sum = achieved_rewards.sum()
    counter = achieved_rewards.numel()

    """
    setup gridpath environment
    """

    env = gym.make("GridWorld-v0", render_mode="rgb_array").unwrapped
    env.reset()
    n_actions = env.action_space.n
    

    # example of screen
    example_screen = get_screen(env)
    # print(example_screen.shape)
    # plt.figure()
    # plt.axis('off')
    # plt.imshow(example_screen.cpu().squeeze(0).permute(1, 2, 0).numpy(),  interpolation='none')
    # plt.show()

    """
    setup policy and target network
    """
    num_channels = 3
    # Create a convolutional network based on arguments and add 1x1 conv layer
    features_net, add_on_layers = get_network(num_channels, args)
    
    policy_net = ProtoTree(num_classes=n_actions, feature_net = features_net, args = args, add_on_layers = add_on_layers)
    policy_net = policy_net.to(device=device)

    # Determine which optimizer should be used to update the tree parameters
    optimizer, params_to_freeze, params_to_train = get_optimizer(policy_net, args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)
    policy_net, epoch = init_tree(policy_net, optimizer, scheduler, device, args)
    
    policy_net.save(f'{log.checkpoint_dir}/tree_init')
    log.log_message("Max depth %s, so %s internal nodes and %s leaves"%(args.depth, policy_net.num_branches, policy_net.num_leaves))
    analyse_output_shape_dqn(policy_net, log, device, example_screen)

    target_net = deepcopy(policy_net)
    target_net = target_net.to(device=device)
    
