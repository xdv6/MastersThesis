# Interpreteerbare besluitvorming in diep versterkend leren met een prototype boomstructuur

Deze repository bevat de code gebruikt bij het schrijven van de masterproef "Interpreteerbare besluitvorming in diep versterkend leren met een prototype boomstructuur", ingediend tot het behalen van de academische graad van Master of Science in de industriële wetenschappen: informatica.

## Benodigdheden

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



## 1) Deep Q-Network



### Cart Pole

De folder ```masteproef/notebooks_cartpole``` bevat een aangepaste versie van de  "Reinforcement Learning (DQN) Tutorial" op Pytorch. Tijdens het ontwikkelen en trainen van de agent, is deze tutorial een tijd neergehaald en uiteindelijk vervangen door een tutorial dat geen gebruik maakt van de pixelspace, maar van observatie-parameters. De versie met pixelspace waaruit vertrokken is, kan nog steeds teruggevonden worden op: https://h-huang.github.io/tutorials/intermediate/reinforcement_q_learning.html.



### Frozen Lake

In het bestand ```masterproef/notebooks_frozen_lake/train_frozen_lake.ipynb``` kan een notebook gevonden worden voor het trainen van de DQN-agent op de Frozen Lake-omgeving. De map bevat ook de notebook ```masterproef/notebooks_frozen_lake/test_frozen_lake.ipynb```. Deze notebook maakt gebruik van de ```HumanRendering``` wrapper en kan gebruikt worden om een getrainde policy in actie te zien. 

De notebook ```masterproef/notebooks_frozen_lake/label_frozen_lake.ipynb``` is gebruikt om een dataset aan te maken waarbij omgevingsbeelden worden gelabeld met de actie voorgesteld door een getrainde policy. 

### Eigen omgeving

In de map ```masterproef/eigen_environm/gym-examples/gym_game``` staat de code voor de eigen aangemaakte omgeving. De code maakt gebruik van een framework voorzien door de gym-library om zelf omgevingen aan te maken en er is vertrokken van volgend voorbeeld:  https://github.com/Farama-Foundation/gym-examples. De logica is geschreven in pygame.

Om de zelf aangemaakte omgeving te integreren in de gym-library moet eerst het script  ```masterproef/eigen_environm/gym-examples/setup.py``` uitgevoerd worden. Dit zorgt ervoor dat de omgeving kan worden aangemaakt  aan de hand van de functie ```gym.make(<env>)```.

De notebooks ```masterproef/eigen_environm/gym-examples/train_gridpath.ipynb```, ```masterproef/eigen_environm/gym-examples/label_gridpath.ipynb``` en ```masterproef/eigen_environm/gym-examples/test_gridpath.ipynb``` worden respectievelijk gebruikt om de policy te trainen, testen en te labelen.

De notebook ```masterproef/eigen_environm/gym-examples/test_env_keyboard.ipynb```  bevat een notebook die gebruikt is om de logica van de aangemaakte omgeving, visueel te testen. Tot slot bevat het bestand ```masterproef/eigen_environm/gym-examples/process_images.ipynb``` code die gebruikt is om de afbeelding op de achtergrond in de omgeving aan te passen. 











