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
from prototree.test import eval, eval_fidelity
from prototree.prune import prune
from prototree.project import project, project_with_class_constraints
from prototree.upsample import upsample

from train_integratie_tree import train_epoch, train_leaves_epoch
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

    # Determine which optimizer should be used to update the policy_net parameters
    optimizer, params_to_freeze, params_to_train = get_optimizer(policy_net, args)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)
    policy_net, epoch = init_tree(policy_net, optimizer, scheduler, device, args)
    
    policy_net.save(f'{log.checkpoint_dir}/tree_init')
    log.log_message("Max depth %s, so %s internal nodes and %s leaves"%(args.depth, policy_net.num_branches, policy_net.num_leaves))
    analyse_output_shape_dqn(policy_net, log, device, example_screen)

    target_net = deepcopy(policy_net)

    leaf_labels = dict()
    best_train_acc = 0.
    best_test_acc = 0.

    """
    Train the policy_net
    """
    # # dummmy state batch
    # memory = ReplayMemory(config.get("REPLAY_BUFFER"))
    
    # state_batch = get_screen(env)
    # state_batch = state_batch.repeat(64, 1, 1, 1)

    # next_state_batch = get_screen(env)
    # next_state_batch = next_state_batch.repeat(64, 1, 1, 1)
    # train_info = train_epoch(state_batch,next_state_batch, config, memory, policy_net, target_net, optimizer, epoch, args.disable_derivative_free_leaf_optim, device, log, log_prefix)
    
    config =  {
    "BATCH_SIZE":64,
    "GAMMA" : 0.999,
    "EPS_START": 1,
    "EPS_END" : 0.1,
    "lr":0.0001, 
    "REPLAY_BUFFER":10000,
    "EPISODES": 1,
    "TARGET_UPDATE": 200,
    "SAVE_FREQ": 10,
    "RESET_ENV_FREQ": 200,
    "DDQN": False,
    "MODEL_dir_file": "./model/stop_border_lagere_lr",
    }

    # Define the custom x axis metric
    # wandb.define_metric("episode")

    # Define which metrics to plot against that x-axis
    # wandb.define_metric("reached_target", step_metric='episode')
    # wandb.define_metric("win_count", step_metric='episode')
    # wandb.define_metric("mean_reward", step_metric='episode')
    # wandb.define_metric("number_of_actions_in_episode", step_metric='episode')

    memory = ReplayMemory(config.get("REPLAY_BUFFER"))

    for epoch in range(config.get("EPISODES")):
        # Initialize the environment and state
        env.reset()

        # state based on patch of screen (3x3 around agent)
        state = get_screen(env)
        spel_gelukt = 0
        
        for t in count():
            env.render()
            # wrapped._render_frame()
            action = select_action(state, policy_net, n_actions, config)
            _, reward, done, _, _ = env.step(action.item())
            
            running_sum += reward
            counter += 1
            mean = running_sum / counter

            reward = torch.tensor([reward], device=device)
            
            if not done:
                next_state = get_screen(env)
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state


            """
            Optimize the policy_net
            """
            if len(memory) > config.get("BATCH_SIZE"):
                # Perform one step of the optimization 
                log.log_message("\nEpoch %s"%str(epoch))
                # Freeze (part of) network for some epochs if indicated in args
                freeze(policy_net, epoch, params_to_freeze, params_to_train, args, log)
                log_learning_rates(optimizer, args, log)
                train_info = train_epoch(config, memory, policy_net, target_net, optimizer, epoch, args.disable_derivative_free_leaf_optim, device, log, log_prefix)
                save_tree(policy_net, optimizer, scheduler, epoch, log, args)
                # best_train_acc = save_best_train_tree(policy_net, optimizer, scheduler, best_train_acc, train_info['train_accuracy'], log)
                leaf_labels = analyse_leafs(policy_net, epoch, n_actions, leaf_labels, args.pruning_threshold_leaves, log)

            # if agent did not reach target after RESET_ENV_FREQ actions, reset environment
            if (t + 1) % config.get("RESET_ENV_FREQ") == 0:
                done = True

            if done:
                if reward == 1000:
                    spel_gelukt = 1
                    win_count += 1

                log_dict = {
                    "episode": epoch + 1,
                    "reached_target": spel_gelukt
                }
                # wandb.log(log_dict)
                # wandb.log({"number_of_actions_in_episode": t})
                # wandb.log({"win_count": win_count})
                # wandb.log({"mean_reward": mean})
                break
            

        # Update the target network, copying all weights and biases to target DQN
        if epoch % config.get("TARGET_UPDATE") == 0:
            target_net = deepcopy(policy_net)
        
        # save model after frequency
        # if epoch % config.get("SAVE_FREQ") == 0:
        #     torch.save(policy_net, config.get("MODEL_dir_file") + str(epoch) + '.pkl')

    print('Complete')
    env.render()
    env.close()


    '''
    PRUNE
    '''
    pruned = prune(policy_net, args.pruning_threshold_leaves, log)
    name = "pruned"
    save_tree_description(policy_net, optimizer, scheduler, name, log)
    pruned_tree = deepcopy(policy_net)
    # Analyse and evaluate pruned policy_net
    leaf_labels = analyse_leafs(policy_net, epoch+2, n_actions, leaf_labels, args.pruning_threshold_leaves, log)
    analyse_leaf_distributions(policy_net, log)


    pruned_tree = policy_net

    # '''
    # PROJECT
    # '''
    # project_info, policy_net = project_with_class_constraints(deepcopy(pruned_tree), projectloader, device, args, log)
    # name = "pruned_and_projected"
    # save_tree_description(policy_net, optimizer, scheduler, name, log)
    # pruned_projected_tree = deepcopy(policy_net)
    # # Analyse and evaluate pruned policy_net with projected prototypes
    # average_distance_nearest_image(project_info, policy_net, log)
    # leaf_labels = analyse_leafs(policy_net, epoch+3, n_actions, leaf_labels, args.pruning_threshold_leaves, log)
    # analyse_leaf_distributions(policy_net, log)
    # eval_info = eval(policy_net, testloader, name, device, log)
    # pruned_projected_test_acc = eval_info['test_accuracy']
    # eval_info_samplemax = eval(policy_net, testloader, name, device, log, 'sample_max')
    # get_avg_path_length(policy_net, eval_info_samplemax, log)
    # eval_info_greedy = eval(policy_net, testloader, name, device, log, 'greedy')
    # get_avg_path_length(policy_net, eval_info_greedy, log)
    # fidelity_info = eval_fidelity(policy_net, testloader, device, log)

    # # Upsample prototype for visualization
    # project_info = upsample(policy_net, project_info, projectloader, name, args, log)
    
    # # visualize policy_net
    # gen_vis(policy_net, name, args, classes)


    
