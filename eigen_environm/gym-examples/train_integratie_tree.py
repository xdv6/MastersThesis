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

from util.log import Log

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

    print(f' Loss: {loss.item():.3f}')
    
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

            batch_memory = memory.memory[i:i+config.get("BATCH_SIZE")]
            state_list = [transition.state for transition in batch_memory]
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


