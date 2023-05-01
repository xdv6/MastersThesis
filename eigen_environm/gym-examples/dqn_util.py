import os
import gym
import gym_game
import pygame
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import pkg_resources
import time
import matplotlib.pyplot as plt
import wandb

import numpy as np
import argparse
import os
import torch
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor, Normalize, Compose, Lambda
from torch.utils.data import TensorDataset, DataLoader

resize = T.Compose([T.ToPILImage(),T.Resize(198, interpolation=Image.CUBIC),T.ToTensor()])
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_screen(env):
    screen = env.render().transpose((2, 0, 1))  # transpose into torch order (CHW)
    _, screen_height, screen_width = screen.shape

     # area around agent
    # coordinaat van linkerbovenhoek rechthoek
    x_pixel_coo_agent = env._agent_location[0] * env.pix_square_size
    y_pixel_coo_agent = env._agent_location[1] * env.pix_square_size

    x_coo_right_up = x_pixel_coo_agent + 2 * env.pix_square_size
    x_coo_right_down = x_pixel_coo_agent - env.pix_square_size

    y_coo_left_down = y_pixel_coo_agent + 2 * env.pix_square_size
    y_coo_left_up = y_pixel_coo_agent - env.pix_square_size

    # left handed coordinate system
    screen = screen[:,y_coo_left_up:y_coo_left_down, x_coo_right_down:x_coo_right_up]

    

    # Convert to float, rescare, convert to torch tensor (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)




class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


    

def select_action(state, policy_net, n_actions, config):
    global steps_done
    sample = random.random()

    # steps_done = -100000 * ln((0.1 - eps_threshold) / 0.9)
    # eps_threshold = config.get("EPS_END") + (config.get("EPS_START") - config.get("EPS_END")) * math.exp(-1. * steps_done / config.get("EPS_DECAY"))
    eps_threshold = config.get("EPS_END")

    wandb.log({"eps_threshold": eps_threshold})
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    



def get_batch(memory, config):
    if len(memory) < config.get("BATCH_SIZE"):
        return
    transitions = memory.sample( config.get("BATCH_SIZE"))
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.uint8)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    return (state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask)




def get_dataloaders_dqn(args, state_batch):
    """
    Get data loaders
    """
    # Obtain the dataset
    trainset, projectset, testset, shape = get_data_dqn(state_batch)
    c, w, h = shape
    # Determine if GPU should be used
    cuda = not args.disable_cuda and torch.cuda.is_available()
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              pin_memory=cuda,
                                              num_workers=1
                                              )
    projectloader = torch.utils.data.DataLoader(projectset,
                                                #    batch_size=args.batch_size,
                                                # make batch size smaller to prevent out of memory errors during projection
                                                # batch_size=int(
                                                #   args.batch_size/4),
                                                batch_size = args.batch_size,
                                                shuffle=False,
                                                pin_memory=cuda,
                                                num_workers=1
                                                )
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=cuda,
                                             num_workers=1
                                             )
    classes = ['down', 'left', 'right', 'up']
    print("Num classes (k) = ", len(classes), flush=True)
    return trainloader, projectloader, testloader, classes, c


def get_data_dqn(state_batch, img_size=198):
    """
    Load the proper dataset based on the parsed arguments
    :param args: The arguments in which is specified which dataset should be used
    :return: a 5-tuple consisting of:
                - The train data set
                - The project data set (usually train data set without augmentation)
                - The test data set
                - a tuple containing all possible class labels
                - a tuple containing the shape (depth, width, height) of the input images
    """

    shape = (3, img_size, img_size)
    trainset = TensorDataset(state_batch)
    projectset = TensorDataset(state_batch)
    testset = TensorDataset(state_batch)

    return trainset, projectset, testset, shape



def analyse_output_shape_dqn(tree, log, device, xs):
    with torch.no_grad():
        xs  = xs.to(device)
        log.log_message("Image input shape: "+str(xs[0,:,:,:].shape))
        log.log_message("Features output shape (without 1x1 conv layer): "+str(tree._net(xs).shape))
        log.log_message("Convolutional output shape (with 1x1 conv layer): "+str(tree._add_on(tree._net(xs)).shape))
        log.log_message("Prototypes shape: "+str(tree.prototype_layer.prototype_vectors.shape))






    







