# Interpreteerbare besluitvorming in diep versterkend leren met een prototype boomstructuur

Deze repository bevat de code gebruikt bij het schrijven van de masterproef "Interpreteerbare besluitvorming in diep versterkend leren met een prototype boomstructuur", ingediend tot het behalen van de academische graad van Master of Science in de industriële wetenschappen: informatica.

## Benodigdheden

* graphviz

- Python 3
- [PyTorch](https://pytorch.org/get-started/locally/) >= 1.5 and <= 1.7!
- Optioneel: CUDA

- numpy
- pandas
- opencv-python
- tqdm
- scipy
- matplotlib
- requests (voor de CARS-dataset)
- gdown ( voor de CUB-dataset)
- gym  (v0.26.2)
- pygame
- wandb 

Voor het loggen van waarden en het genereren van plots is er gebruik gemaakt van wandb.  (https://wandb.ai/). 

De verschillende notebooks en scripts voor het trainen van modellen bevatten:  ```run = wandb.init(project="refactor", entity="xdvisch", config=config)```. Hierbij moet entity aangepast worden  om nadien de eigen API key  mee tegeven. Meer informatie over het aanmaken van account, initialisatie en API key kan gevonden worden op: https://docs.wandb.ai/quickstart.

## 1) Deep Q-Network



### Cart Pole

De folder ```masteproef/notebooks_cartpole``` bevat een aangepaste versie van de  "Reinforcement Learning (DQN) Tutorial" op Pytorch. Tijdens het ontwikkelen en trainen van de agent, is deze tutorial een tijd neergehaald en uiteindelijk vervangen door een tutorial dat geen gebruik maakt van de pixelspace, maar van observatie-parameters. De versie met pixelspace waaruit vertrokken is, kan nog steeds teruggevonden worden op: https://h-huang.github.io/tutorials/intermediate/reinforcement_q_learning.html.



### Frozen Lake

In het bestand ```masterproef/notebooks_frozen_lake/train_frozen_lake.ipynb``` kan een notebook gevonden worden voor het trainen van de DQN-agent op de Frozen Lake-omgeving. De map bevat ook de notebook ```masterproef/notebooks_frozen_lake/test_frozen_lake.ipynb```. Deze notebook maakt gebruik van de ```HumanRendering``` wrapper en kan gebruikt worden om een getrainde policy in actie te zien. 

De notebook ```masterproef/notebooks_frozen_lake/label_frozen_lake.ipynb``` is gebruikt om een dataset aan te maken waarbij omgevingsbeelden worden gelabeld met de actie voorgesteld door een getrainde policy. 

### Eigen omgeving

In de map ```masterproef/eigen_environm/gym-examples/gym_game``` staat de code voor de eigen aangemaakte omgeving. De code maakt gebruik van een framework voorzien door de gym-library om zelf omgevingen aan te maken en er is vertrokken van volgend voorbeeld:  https://github.com/Farama-Foundation/gym-examples. De logica is geschreven in pygame.

De notebooks ```masterproef/eigen_environm/gym-examples/train_gridpath.ipynb```, ```masterproef/eigen_environm/gym-examples/label_gridpath.ipynb``` en ```masterproef/eigen_environm/gym-examples/test_gridpath.ipynb``` worden respectievelijk gebruikt om de policy te trainen, testen en te labelen.

De notebook ```masterproef/eigen_environm/gym-examples/test_env_keyboard.ipynb```  bevat een notebook die gebruikt is om de logica van de aangemaakte omgeving visueel te testen. Tot slot bevat het bestand ```masterproef/eigen_environm/gym-examples/process_images.ipynb``` code die gebruikt is om de afbeelding op de achtergrond in de omgeving aan te passen. 



​		

## 2) ProtoTree

De map ```masterproef/ProtoTree-main``` bevat een gekloonde versie van de ProtoTree (https://github.com/M-Nauta/ProtoTree?utm_source=catalyzex.com). De code is aangepast zodat het de mogelijkheid heeft om getraind te worden op omgevingsbeelden van Frozen Lake en van de eigen aangemaakte omgeving. Om dit te doen moet respectievelijk de optie ```--dataset ``` op ```frozen_lake``` en ```gridpath``` geplaatst worden. 

De code is ook licht aangepast voor de integratie met de DQN (zie 3) , indien de code wordt gebruikt voor het supervised trainen met de optie ```--disable_derivative_free_leaf_optim``` (geeft mindere resultaten) moeten er 3 bestanden licht aangepast worden: 

1)  In ```masterproef/ProtoTree-main/util/visualize.py ``` en ```masterproef/ProtoTree-main/prototree/prune.py``` moet elke voorkomst van:

   ```py
   F.softmax(node.distribution() - torch.max(node.distribution()), dim=0)
   # of 
   F.softmax(leaf.distribution() - torch.max(leaf.distribution()), dim=0)
   ```
   
   respectievelijk vervangen worden door:

   ```py
   node.distribution()
   # of
   leaf.distribution()	
   ```

2. In ``` masterproef/ProtoTree-main/prototree/leaf.py ``` op lijn 75 moet ``` return self._dist_params``` vervangen worden door ```return F.softmax(self._dist_params - torch.max(self._dist_params), dim=0)```

   

## 3) Integratie DQN en ProtoTree

Het script ```masterproef/eigen_environm/gym-examples/dqn_prototree_integratie.py``` bevat de code voor het trainen van de DQN-agent op de eigen aangemaakte omgeving, waarbij er gebruik wordt gemaakt van de ProtoTree in plaats van het eenvoudige CNN zoals bij 1). 

Het script maakt ook gebruik van de hulpbestanden ```masterproef/eigen_environm/gym-examples/train_integratie_tree.py``` en ```masterproef/eigen_environm/gym-examples/dqn_integratie_util.py```.  Het script ```masterproef/eigen_environm/gym-examples/train_integratie_tree.py``` bevat aangepaste code voor het trainingsproces van de ProtoTree.  De code in ```masterproef/eigen_environm/gym-examples/dqn_integratie_util.py``` is een aangepaste versie van de DQN hulpklassen en methoden zodat deze gebruikt kunnen worden om te interageren met de ProtoTree.



Het script kan uitgevoerd worden ( in de directory ```masterproef/eigen_environm/gym-examples/``` ) aan de hand van onderstaand commando: 

```python3 dqn_prototree_integratie.py --epochs 60 --lr 0.01 --lr_block 0.01 --num_features 3 --depth 2 --net vgg11 --pruning_threshold_leaves 0.4  --batch_size 64 --log_dir ./runs/refactor --milestones 30,50,60,70 --disable_derivative_free_leaf_optim --lr_pi 0.001```

Alle argumenten die weergegeven staan kunnen aangepast worden buiten ```--net vgg11``` en ```--disable_derivative_free_leaf_optim```. De batch_size moet ook een meervoud zijn van 32.  
