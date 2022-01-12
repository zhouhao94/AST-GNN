# AST-GNN: An attention-based spatio-temporal graph neural network for Interaction-aware pedestrian trajectory prediction
### Hao Zhou, Dongchun Ren, Huaxia Xia, Mingyu Fan, Xu Yang, Hai Huang <br />

### Citaion
You can cite our paper using: 
```
@article{zhou2021ast,
  title={AST-GNN: An attention-based spatio-temporal graph neural network for Interaction-aware pedestrian trajectory prediction},
  author={Zhou, Hao and Ren, Dongchun and Xia, Huaxia and Fan, Mingyu and Yang, Xu and Huang, Hai},
  journal={Neurocomputing},
  volume={445},
  pages={298--308},
  year={2021},
  publisher={Elsevier}
}
```

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
To train a model for each data set with the best configuration as in the paper, simply run:
```bash
./train.sh  
```

To use the pretrained models at `checkpoint/` and evaluate the models performance run:
```bash
test.py
```
Note-1: The initiation of training and testing might take a while as the code creates a cache to store the spatio-temporal graphs.

Note-2: The code is partially based on Social-STGCNN (https://github.com/abduallahmohamed/Social-STGCNN).
