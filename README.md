# Interpretable deep reinforcement learning policies using prototype trees

This repository contains the code used in the writing of the master's thesis titled "Interpretable deep reinforcement learning policies using prototype trees" submitted to obtain the academic degree of Master of Science in Information Engineering Technology.

## Requirements

* graphviz
* Python 3
* PyTorch >= 1.5 and <= 1.7! (https://pytorch.org/get-started/locally/)
* Optional: CUDA

Python packages:
* numpy
* pandas
* opencv-python
* tqdm
* scipy
* matplotlib
* requests (for the CARS dataset)
* gdown (for the CUB dataset)
* gym (v0.26.2)
* pygame
* wandb

For logging values and generating plots, wandb was used (https://wandb.ai/). The different notebooks and scripts for training models include the line `run = wandb.init(project="refactor", entity="xdvisch", config=config)`. The entity should be modified to provide your own API key. More information on creating an account, initialization, and API key can be found at: https://docs.wandb.ai/quickstart.

## 1) Deep Q-Network

### Cart Pole

The folder `masterproef/notebooks_cartpole` contains a modified version of the "Reinforcement Learning (DQN) Tutorial" in PyTorch. During the development and training of the agent, this tutorial was temporarily taken down and eventually replaced with a tutorial that does not use pixel space but observation parameters. The version with pixel space that was used as a starting point can still be found at: https://h-huang.github.io/tutorials/intermediate/reinforcement_q_learning.html.

### Frozen Lake

In the file `masterproef/notebooks_frozen_lake/train_frozen_lake.ipynb`, you can find a notebook for training the DQN agent on the Frozen Lake environment. The folder also contains the notebook `masterproef/notebooks_frozen_lake/test_frozen_lake.ipynb`. This notebook uses the `HumanRendering` wrapper and can be used to observe a trained policy in action.

The notebook `masterproef/notebooks_frozen_lake/label_frozen_lake.ipynb` was used to create a dataset where environment images are labeled with the action suggested by a trained policy.

### Custom Environment

In the folder `masterproef/eigen_environm/gym-examples/gym_game`, you'll find the code for the custom environment created. The code uses a framework provided by the gym library to create custom environments, and it was based on the following example: https://github.com/Farama-Foundation/gym-examples. The logic is written in pygame.

The notebooks `masterproef/eigen_environm/gym-examples/train_gridpath.ipynb`, `masterproef/eigen_environm/gym-examples/label_gridpath.ipynb`, and `masterproef/eigen_environm/gym-examples/test_gridpath.ipynb` are used for training, labeling, and testing the policy, respectively.

The file `masterproef/eigen_environm/gym-examples/test_env_keyboard.ipynb` contains a notebook that was used to visually test the logic of the custom environment. Finally, the file `masterproef/eigen_environm/gym-examples/process_images.ipynb` contains code that was used to modify the background image in the environment.

## 2) ProtoTree

The folder `masterproef/ProtoTree-main` contains a cloned version of ProtoTree (https://github.com/M-Nauta/ProtoTree?utm_source=catalyzex.com). The code has been modified to allow training on Frozen Lake environment images and custom environment images. To do this, the `--dataset` option should be set to either `frozen_lake` or `gridpath`.

The code has also been slightly modified for integration with DQN (see section 3). If the code is used for supervised training, three files need to be modified:

1) In `masterproef/ProtoTree-main/util/visualize.py` and `masterproef/ProtoTree-main/prototree/prune.py`, every occurrence of:

   ```py
   F.softmax(node.distribution() - torch.max(node.distribution()), dim=0)
   # or
   F.softmax(leaf.distribution() - torch.max(leaf.distribution()), dim=0)
   ```
   
   should be replaced with:

   ```py
   node.distribution()
   # of
   leaf.distribution()	
   ```

2. In masterproef/ProtoTree-main/prototree/leaf.py, line 75 should be changed from return self._dist_params to return F.softmax(self._dist_params - torch.max(self._dist_params), dim=0).

   

## 3) Integration of DQN and ProtoTree

The script masterproef/eigen_environm/gym-examples/dqn_prototree_integratie.py contains the code for training the DQN agent on the custom environment using ProtoTree instead of a simple CNN as in section 1.

The script also uses the helper files masterproef/eigen_environm/gym-examples/train_integratie_tree.py and masterproef/eigen_environm/gym-examples/dqn_integratie_util.py. The script masterproef/eigen_environm/gym-examples/train_integratie_tree.py contains modified code for the ProtoTree training process. The code in masterproef/eigen_environm/gym-examples/dqn_integratie_util.py is an adapted version of the DQN helper classes and methods, allowing interaction with ProtoTree.

The script can be executed (in the directory masterproef/eigen_environm/gym-examples/) using the following command:

```python3 dqn_prototree_integratie.py --epochs 60 --lr 0.01 --lr_block 0.01 --num_features 3 --depth 2 --net vgg11 --pruning_threshold_leaves 0.4 --batch_size 64 --log_dir ./runs/refactor --milestones 30,50,60,70 --disable_derivative_free_leaf_optim --lr_pi 0.001
```

All the arguments listed can be adjusted except for --net vgg11 and --disable_derivative_free_leaf_optim. The batch size should also be a multiple of 32.
