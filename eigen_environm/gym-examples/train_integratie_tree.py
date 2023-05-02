from tqdm import tqdm
import argparse
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from dqn_util import *
from prototree.prototree import ProtoTree


import os
import cv2
import argparse
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from util.log import Log

from util.log import Log
import wandb

# Create a transformation to convert the tensor to a PIL Image
transform = transforms.ToPILImage()

# def train_epoch_kontschieder(policy_net: ProtoTree,
#                 train_loader: DataLoader,
#                 optimizer: torch.optim.Optimizer,
#                 epoch: int,
#                 disable_derivative_free_leaf_optim: bool,
#                 device,
#                 log: Log = None,
#                 log_prefix: str = 'log_train_epochs',
#                 progress_prefix: str = 'Train Epoch'
#                 ) -> dict:

#     policy_net = policy_net.to(device)

#     # Store info about the procedure
#     train_info = dict()
#     total_loss = 0.
#     total_acc = 0.

#     # Create a log if required
#     log_loss = f'{log_prefix}_losses'
#     if log is not None and epoch==1:
#         log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')
    
#     # Reset the gradients
#     optimizer.zero_grad()

#     if disable_derivative_free_leaf_optim:
#         print("WARNING: kontschieder arguments will be ignored when training leaves with gradient descent")
#     else:
#         if policy_net._kontschieder_normalization:
#             # Iterate over the dataset multiple times to learn leaves following Kontschieder's approach
#             for _ in range(10):
#                 # Train leaves with derivative-free algorithm using normalization factor
#                 train_leaves_epoch(policy_net, train_loader, epoch, device)
#         else:
#             # Train leaves with Kontschieder's derivative-free algorithm, but using softmax
#             train_leaves_epoch(policy_net, train_loader, epoch, device)
#     # Train prototypes and network. 
#     # If disable_derivative_free_leaf_optim, leafs are optimized with gradient descent as well.
#     # Show progress on progress bar
#     # train_iter = tqdm(enumerate(train_loader),
#     #                     total=len(train_loader),
#     #                     desc=progress_prefix+' %s'%epoch,
#     #                     ncols=0)
#     # Make sure the model is in train mode
#     policy_net.train()
#     for i, (xs, ys) in train_iter:
#         xs, ys = xs.to(device), ys.to(device)

#         # Reset the gradients
#         optimizer.zero_grad()
#         # Perform a forward pass through the network
#         ys_pred, _ = policy_net.forward(xs)
#         # Compute the loss
#         if policy_net._log_probabilities:
#             loss = F.nll_loss(ys_pred, ys)
#         else:
#             loss = F.nll_loss(torch.log(ys_pred), ys)
#         # Compute the gradient
#         loss.backward()
#         # Update model parameters
#         optimizer.step()

#         # Count the number of correct classifications
#         ys_pred = torch.argmax(ys_pred, dim=1)
        
#         correct = torch.sum(torch.eq(ys_pred, ys))
#         acc = correct.item() / float(len(xs))

#         train_iter.set_postfix_str(
#             f'Batch [{i + 1}/{len(train_loader)}], Loss: {loss.item():.3f}, Acc: {acc:.3f}'
#         )
#         # Compute metrics over this batch
#         total_loss+=loss.item()
#         total_acc+=acc

#         if log is not None:
#             log.log_values(log_loss, epoch, i + 1, loss.item(), acc)
        
#     train_info['loss'] = total_loss/float(i+1)
#     train_info['train_accuracy'] = total_acc/float(i+1)
#     return train_info 

# # Updates leaves with derivative-free algorithm
# def train_leaves_epoch(policy_net: ProtoTree,
#                         train_loader: DataLoader,
#                         epoch: int,
#                         device,
#                         progress_prefix: str = 'Train Leafs Epoch'
#                         ) -> dict:

#     #Make sure the policy_net is in eval mode for updating leafs
#     policy_net.eval()

#     with torch.no_grad():
#         _old_dist_params = dict()
#         for leaf in policy_net.leaves:
#             _old_dist_params[leaf] = leaf._dist_params.detach().clone()
#         # Optimize class distributions in leafs
#         eye = torch.eye(policy_net._num_classes).to(device)

#         # Show progress on progress bar
#         # train_iter = tqdm(enumerate(train_loader),
#         #                 total=len(train_loader),
#         #                 desc=progress_prefix+' %s'%epoch,
#         #                 ncols=0)
        
        
#         # Iterate through the data set
#         update_sum = dict()

#         # Create empty tensor for each leaf that will be filled with new values
#         for leaf in policy_net.leaves:
#             update_sum[leaf] = torch.zeros_like(leaf._dist_params)
        
#         for i, (xs, ys) in train_iter:
#             xs, ys = xs.to(device), ys.to(device)
#             #Train leafs without gradient descent
#             out, info = policy_net.forward(xs)
#             target = eye[ys] #shape (batchsize, num_classes) 
#             for leaf in policy_net.leaves:  
#                 if policy_net._log_probabilities:
#                     # log version
#                     update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - out, dim=0))
#                 else:
#                     update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/out, dim=0)
#                 update_sum[leaf] += update

#         for leaf in policy_net.leaves:
#             leaf._dist_params -= leaf._dist_params #set current dist params to zero
#             leaf._dist_params += update_sum[leaf] #give dist params new value


def train_epoch(config: dict,
                memory: ReplayMemory,
                policy_net: ProtoTree,
                target_net: ProtoTree,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                disable_derivative_free_leaf_optim: bool,
                device,
                log: Log = None,
                log_prefix: str = 'log_train_epochs',
                progress_prefix: str = 'Train Epoch'
                ) -> dict:
    
    
    policy_net = policy_net.to(device)
    # Make sure the model is in eval mode
    policy_net.eval()
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.
    # Create a log if required
    log_loss = f'{log_prefix}_losses'

    nr_batches = 1

    with torch.no_grad():
        _old_dist_params = dict()
        for leaf in policy_net.leaves:
            _old_dist_params[leaf] = leaf._dist_params.detach().clone()
        # Optimize class distributions in leafs
        eye = torch.eye(policy_net._num_classes).to(device)

    # Iterate through the data set to update leaves, prototypes and network
    state_batch, action_batch, reward_batch, non_final_next_states, non_final_mask = get_batch(memory, config)

    xs = state_batch
    ys = non_final_next_states
    # Make sure the model is in train mode
    policy_net.train()
    # Reset the gradients
    optimizer.zero_grad()

    xs, ys = xs.to(device), ys.to(device)

    # Perform a forward pass through the policy network
    ys_pred, info = policy_net.forward(xs)
    # compute the Q-estimate 
    state_action_values = ys_pred.gather(1, action_batch)

    # Perform a forward pass through the target network
    with torch.no_grad():
        ys_target, _ = target_net.forward(ys)
        next_state_values = torch.zeros( config.get("BATCH_SIZE"), device=device)
        # compute the Q-target
        next_state_values[non_final_mask] = ys_target.max(1)[0]

    expected_state_action_values = ((next_state_values * config.get("GAMMA") + reward_batch).unsqueeze(1))
    

    # Learn prototypes and network with gradient descent. 
    # If disable_derivative_free_leaf_optim, leaves are optimized with gradient descent as well.
    # Compute the loss

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    
    # Compute the gradient
    loss.backward()
    # Update model parameters
    optimizer.step()
    
    if not disable_derivative_free_leaf_optim:
        #Update leaves with derivate-free algorithm
        #Make sure the policy_net is in eval mode
        policy_net.eval()
        with torch.no_grad():
            target = eye[ys] #shape (batchsize, num_classes) 
            for leaf in policy_net.leaves:  
                if policy_net._log_probabilities:
                    # log version
                    update = torch.exp(torch.logsumexp(info['pa_tensor'][leaf.index] + leaf.distribution() + torch.log(target) - ys_pred, dim=0))
                else:
                    update = torch.sum((info['pa_tensor'][leaf.index] * leaf.distribution() * target)/ys_pred, dim=0)  
                leaf._dist_params -= (_old_dist_params[leaf]/nr_batches)
                F.relu_(leaf._dist_params) #dist_params values can get slightly negative because of floating point issues. therefore, set to zero.
                leaf._dist_params += update

    # print(f' Loss: {loss.item():.3f}')
    wandb.log({"loss": loss})
    
    # Compute metrics over this batch
    total_loss+=loss.item()

    train_info['loss'] = total_loss/float(1)
    return train_info 













def project_dqn_tree(config: dict,
            memory: ReplayMemory,
            tree: ProtoTree,
            device,
            args: argparse.Namespace,
            log: Log,  
            log_prefix: str = 'log_projection',  # TODO
            progress_prefix: str = 'Projection'
            ) -> dict:
        
    log.log_message("\nProjecting prototypes to nearest training patch (without class restrictions)...")
    # Set the model to evaluation mode
    tree.eval()
    torch.cuda.empty_cache()
    # The goal is to find the latent patch that minimizes the L2 distance to each prototype
    # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
    # Also store info about the image that was used for projection
    global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
    global_min_patches = {j: None for j in range(tree.num_prototypes)}
    global_min_info = {j: None for j in range(tree.num_prototypes)}

    # Get the shape of the prototypes
    W1, H1, D = tree.prototype_shape

    
    
    with torch.no_grad():
        for i in range(0, config.get("REPLAY_BUFFER") - config.get("BATCH_SIZE") + 1, config.get("BATCH_SIZE")):

            all_memory = memory.memory[i:i+config.get("BATCH_SIZE")]
            state_list = [transition.state for transition in all_memory]
            state_tensor = torch.squeeze(torch.stack(state_list), dim=1)

            xs = state_tensor
            xs = xs.to(device)
            # Get the features and distances
            # - features_batch: features tensor (shared by all prototypes)
            #   shape: (batch_size, D, W, H)
            # - distances_batch: distances tensor (for all prototypes)
            #   shape: (batch_size, num_prototypes, W, H)
            # - out_map: a dict mapping decision nodes to distances (indices)
            features_batch, distances_batch, out_map = tree.forward_partial(xs)

            # Get the features dimensions
            bs, D, W, H = features_batch.shape

            # Get a tensor containing the individual latent patches
            # Create the patches by unfolding over both the W and H dimensions
            # TODO -- support for strides in the prototype layer? (corresponds to step size here)
            patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)

            # Iterate over all decision nodes/prototypes
            for node, j in out_map.items():

                # Iterate over all items in the batch
                # Select the features/distances that are relevant to this prototype
                # - distances: distances of the prototype to the latent patches
                #   shape: (W, H)
                # - patches: latent patches
                #   shape: (D, W, H, W1, H1)
                for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):

                    # Find the index of the latent patch that is closest to the prototype
                    min_distance = distances.min()
                    min_distance_ix = distances.argmin()
                    # Use the index to get the closest latent patch
                    closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]

                    # Check if the latent patch is closest for all data samples seen so far
                    if min_distance < global_min_proto_dist[j]:
                        global_min_proto_dist[j] = min_distance
                        global_min_patches[j] = closest_patch
                        global_min_info[j] = {
                            'input_image_ix': i * config.get("BATCH_SIZE") + batch_i,
                            'patch_ix': min_distance_ix.item(),  # Index in a flattened array of the feature map
                            'W': W,
                            'H': H,
                            'W1': W1,
                            'H1': H1,
                            'distance': min_distance.item(),
                            'nearest_input': torch.unsqueeze(xs[batch_i],0),
                            'node_ix': node.index,
                        }


            del features_batch
            del distances_batch
            del out_map
        # Copy the patches to the prototype layer weights
        projection = torch.cat(tuple(global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)),
                                dim=0,
                                out=tree.prototype_layer.prototype_vectors)
        del projection

    return global_min_info, tree




def upsample_with_dqn(tree: ProtoTree, project_info: dict, memory: ReplayMemory, folder_name: str, args: argparse.Namespace, log: Log):
    dir = os.path.join(os.path.join(args.log_dir, args.dir_for_saving_images), folder_name)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with torch.no_grad():
        sim_maps, project_info = get_similarity_maps(tree, project_info, log)
        log.log_message("\nUpsampling prototypes for visualization...")

        all_memory = memory.memory
        state_list = [transition.state for transition in all_memory]
        # shape: [buffer_size, C, H, W]
        state_tensor = torch.squeeze(torch.stack(state_list), dim=1)


        for node, j in tree._out_map.items():
            if node in tree.branches: #do not upsample when node is pruned
                prototype_info = project_info[j]
                decision_node_idx = prototype_info['node_ix']


                x = transform(state_tensor[prototype_info['input_image_ix']]).convert('RGB')
                x.save(os.path.join(dir,'%s_original_image.png'%str(decision_node_idx)))
                    
                x_np = np.asarray(x)
                x_np = np.float32(x_np)/ 255
                if x_np.ndim == 2: #convert grayscale to RGB
                    x_np = np.stack((x_np,)*3, axis=-1)
                
                img_size = x_np.shape[:2]
                similarity_map = sim_maps[j]

                rescaled_sim_map = similarity_map - np.amin(similarity_map)
                rescaled_sim_map= rescaled_sim_map / np.amax(rescaled_sim_map)
                similarity_heatmap = cv2.applyColorMap(np.uint8(255*rescaled_sim_map), cv2.COLORMAP_JET)
                similarity_heatmap = np.float32(similarity_heatmap) / 255
                similarity_heatmap = similarity_heatmap[...,::-1]
                plt.imsave(fname=os.path.join(dir,'%s_heatmap_latent_similaritymap.png'%str(decision_node_idx)), arr=similarity_heatmap, vmin=0.0,vmax=1.0)

                upsampled_act_pattern = cv2.resize(similarity_map,
                                                    dsize=(img_size[1], img_size[0]),
                                                    interpolation=cv2.INTER_CUBIC)
                rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
                rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
                heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_pattern), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[...,::-1]
                
                overlayed_original_img = 0.5 * x_np + 0.2 * heatmap
                plt.imsave(fname=os.path.join(dir,'%s_heatmap_original_image.png'%str(decision_node_idx)), arr=overlayed_original_img, vmin=0.0,vmax=1.0)

                # save the highly activated patch
                masked_similarity_map = np.zeros(similarity_map.shape)
                prototype_index = prototype_info['patch_ix']
                W, H = prototype_info['W'], prototype_info['H']
                assert W == H
                masked_similarity_map[prototype_index // W, prototype_index % W] = 1 #mask similarity map such that only the nearest patch z* is visualized
                
                upsampled_prototype_pattern = cv2.resize(masked_similarity_map,
                                                    dsize=(img_size[1], img_size[0]),
                                                    interpolation=cv2.INTER_CUBIC)
                plt.imsave(fname=os.path.join(dir,'%s_masked_upsampled_heatmap.png'%str(decision_node_idx)), arr=upsampled_prototype_pattern, vmin=0.0,vmax=1.0) 
                    
                high_act_patch_indices = find_high_activation_crop(upsampled_prototype_pattern, args.upsample_threshold)
                high_act_patch = x_np[high_act_patch_indices[0]:high_act_patch_indices[1],
                                                    high_act_patch_indices[2]:high_act_patch_indices[3], :]
                plt.imsave(fname=os.path.join(dir,'%s_nearest_patch_of_image.png'%str(decision_node_idx)), arr=high_act_patch, vmin=0.0,vmax=1.0)

                # save the original image with bounding box showing high activation patch
                imsave_with_bbox(fname=os.path.join(dir,'%s_bounding_box_nearest_patch_of_image.png'%str(decision_node_idx)),
                                    img_rgb=x_np,
                                    bbox_height_start=high_act_patch_indices[0],
                                    bbox_height_end=high_act_patch_indices[1],
                                    bbox_width_start=high_act_patch_indices[2],
                                    bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
    return project_info



def get_similarity_maps(tree: ProtoTree, project_info: dict, log: Log = None):
    log.log_message("\nCalculating similarity maps (after projection)...")
    
    sim_maps = dict()
    for j in project_info.keys():
        nearest_x = project_info[j]['nearest_input']
        with torch.no_grad():
            _, distances_batch, _ = tree.forward_partial(nearest_x)
            sim_maps[j] = torch.exp(-distances_batch[0,j,:,:]).cpu().numpy()
        del nearest_x
        del project_info[j]['nearest_input']
    return sim_maps, project_info


# copied from protopnet
def find_high_activation_crop(mask,threshold):
    threshold = 1.-threshold
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > threshold:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > threshold:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > threshold:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > threshold:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1

# copied from protopnet
def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imshow(img_rgb_float)
    #plt.axis('off')
    plt.imsave(fname, img_rgb_float)