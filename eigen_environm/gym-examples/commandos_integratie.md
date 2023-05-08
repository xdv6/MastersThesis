# Commando's


### main integratie args

```python3 dqn_prototree_integratie.py --epochs 1 --lr 0.01 --lr_block 0.01 --num_features 3 --depth 2 --net vgg11 --pruning_threshold_leaves 0.4  --batch_size 64 --log_dir ./runs/refactor --milestones 30,50,60,70 --disable_derivative_free_leaf_optim```

### Job scheduling gpulab

```dockerfile
# Start from a core gpu-enabled stack version (pytorch)
FROM gitlab.ilabt.imec.be:4567/ilabt/gpu-docker-stacks/tensorflow-notebook:latest

#######################################
## Installing Extra software via apt ##
#######################################

# Apt requires root permissions, so switch to root
USER root

# Install graphiz
RUN apt-get update && \
    apt-get install \
    graphviz 

# Install pip packages
RUN pip install pygame
RUN pip install gym
RUN pip install wandb
RUN pip install opencv-python

# Restore the correct user
USER xdvischu

# change working directory
WORKDIR "/masterproef/eigen_environm/gym-examples"

# login to wandb
RUN cat secret.txt | wandb login --machine

# Copy the script to the container
COPY dqn_prototree_integratie.py .

# Run the script with the specified arguments
CMD ["python", "dqn_prototree_integratie.py", "--epochs", "2000", "--dataset", "gridpath", "--lr", "0.01", "--lr_block", "0.01", "--lr_net", "1e-2", "--num_features", "3", "--depth", "3", "--net", "vgg11", "--freeze_epochs", "2000", "--pruning_threshold_leaves", "0.4", "--disable_derivative_free_leaf_optim", "--batch_size", "64"]

```



