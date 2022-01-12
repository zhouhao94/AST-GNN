# AST-GNN: An attention-based spatio-temporal graph neural network for Interaction-aware pedestrian trajectory prediction
### Hao Zhou, Dongchun Ren, Huaxia Xia, Mingyu Fan, Xu Yang, Hai Huang <br />

### Citaion
You can cite our paper using: 
```
@InProceedings{Mohamed_2020_CVPR,
author = {Mohamed, Abduallah and Qian, Kun and Elhoseiny, Mohamed and Claudel, Christian},
title = {Social-STGCNN: A Social Spatio-Temporal Graph Convolutional Neural Network for Human Trajectory Prediction},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```


#### Model
<div align='center'>
<img src="images/social-stgcnn-model.jpg"></img>
</div>
<br />
Social-STGCNN model consists of 2 building blocks: <br />
1- ST-GCNN: A Spatio-Tempral Graph CNN that creates a spatio-temporal graph embedding representing the previous pedestrians trajectories. <br />
2- TXP-CNN: A Time-Extrapolator CNN that utilizes the spatio-temporal graph embedding to predict future trajectories.<br />

More details are in this description video:<br /> 
<div align='center'>
  
[![Social-STGCNN description](https://img.youtube.com/vi/Yi9rVHJM8WY/0.jpg)](https://www.youtube.com/watch?v=Yi9rVHJM8WY)
</div>

#### Predictions sample

<div align='center'>
<img src="images/social-stgcnn-pred.gif"></img>
</div>
<br />

### Setup: 
The code was written using python 3.6. 
The following libraries are the minimal to run the code: 
```python
import pytorch
import networkx
import numpy
import tqdm
```
or you can have everything set up by running: 
```bash
pip install -r requirements.txt
```
### Using the code:
To use the pretrained models at `checkpoint/` and evaluate the models performance run:
```bash
test.py
```

To train a model for each data set with the best configuration as in the paper, simply run:
```bash
./train.sh  
```
Please note: The initiation of training and testing might take a while as the code creates a cache to store the spatio-temporal graphs.

