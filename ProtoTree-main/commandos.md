# Commando's



## ProtoTypeTree

### ProtoTree trainen (aangepaste dir om te testen ingeschakeld):

```python main_tree.py --epochs 100 --log_dir ./runs/test_CUB_dir --dataset CUB-200-2011 --lr 0.001 --lr_block 0.001 --lr_net 1e-5 --num_features 256 --depth 9 --net resnet50_inat --freeze_epochs 30 --milestones 60,70,80,90,100```

### Local explanation maken

```python main_explain_local.py --log_dir ./runs/protoree_cub --dataset CUB-200-2011 --sample_dir ./data/CUB_200_2011/dataset/test_full/017.Cardinal/Cardinal_0001_17057.jpg  --prototree ./runs/protoree_cub/checkpoints/pruned_and_projected```

### ProtoTree trainen frozen_lake 

```python main_tree.py --epochs 100 --log_dir ./runs/test_frozen_dummy_W1_100 --dataset frozen_lake --lr 0.001 --lr_block 0.001 --lr_net 1e-5 --num_features 3 --depth 4 --net vgg11 --freeze_epochs 30 --milestones 60,70,80,90,100 --W1 100 --H1 100 ```

### reeds getrainde ProtoTree evalueren frozen_lake 

``` python main_tree.py --epochs 1 --log_dir ./runs/test_frozen_dummy_W1_100 --dataset frozen_lake --lr 0.001 --lr_block 0.001 --lr_net 1e-5 --num_features 3 --depth 4 --net vgg11 --freeze_epochs 30 --milestones 60,70,80,90,100 --W1 100 --H1 100 --state_dict_dir_tree ./runs/test_test_frozen_dummy_W1_100/checkpoints/pruned_and_projected```







```text
opmerkingen:
gebruikt geen default num_features want dan krijg je alleen witte kotjes als prototypes
```





## tar compressie

### Compress an entire directory by running:

```tar -zcvf test_frozen.tar.gz ./test_frozen```

### in windows cmd:

```tar -xvzf test_frozen.tar.gz -C D:\master\masterproef\master_thesis\repo\masterproef\ProtoTree-main\runs```





## dot file omzetten naar pdf (in wsl of op gpulab na graphiz)

### graphiz installeren

```
sudo apt install graphviz
```

### dot file effectief omzetten

```dot -Tpdf -Gmargin=0 ./runs/test_frozen/pruned_and_projected/treevis.dot -o ./runs/test_frozen/pruned_and_projected/treevis.pdf```

