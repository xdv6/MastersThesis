# Commando's


### main integratie args

```python main_tree.py --epochs 100 --log_dir ./runs/test_shape --dataset gridpath --lr 0.01 --lr_block 0.01 --lr_net 1e-2 --num_features 3 --depth 4 --net vgg11 --milestones 60,70,80,90,100 --freeze_epochs 100 --pruning_threshold_leaves 0.4 --log_probabilities --disable_derivative_free_leaf_optim```