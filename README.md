## NC-Net for Nuclei Instance Segmentation

Official implementation of Nuclei Probability and Centroid Map Network (NC-Net) for nuclei instance segmentation in histology images. The paper can be read [Here](https://link.springer.com/article/10.1007/s00521-023-08503-2).

### Setup Enviorment
For our experiments we have been using anaconda package manager and we have added environment.yml for creating the same environment that we have used. For setup of enviorment please run the following command:
```console
conda env create -f environment.yml
```

### Datasets
We have trained the model on three Datasets that are CoNSeP, PanNuke and Lizard. The links to the Dataset Download pages are:
- [CoNSeP](https://warwick.ac.uk/fac/cross_fac/tia/data/hovernet/)
- [PanNuke](https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke)
- [Lizard](https://warwick.ac.uk/fac/cross_fac/tia/data/lizard)

```
PWD
|--environment.yml 
|
|--checkpoints
|
|--run_logs
|
|--src
|
|  # please either create it or symlink it
|--data 
   |
   |--consep
   |  |--train
   |      |--001.npy
   |      |--002.npy
   |      |--*.npy
   |      |--n.npy
   |  |--test
   |      |--001.npy
   |      |--002.npy
   |      |--*.npy
   |      |--n.npy
   |
   |--pannuke
   |  |--train
   |      |--001.npy
   |      |--002.npy
   |      |--*.npy
   |      |--n.npy
   |  |--test
   |      |--001.npy
   |      |--002.npy
   |      |--*.npy
   |      |--n.npy
   |
   |--lizard
   |  |--train
   |      |--001.npy
   |      |--002.npy
   |      |--*.npy
   |      |--n.npy
   |  |--test
   |      |--001.npy
   |      |--002.npy
   |      |--*.npy
   |      |--n.npy
   |

```

### Model Training
Before starting model training go through [config.py](./config.py) for setting up various hyper-parameters that are esseintial for the model training. To start training please run the command below:
```console
python train.py
```
> **Note**: The training job currently only works with single GPU and does not support CPU training.

### Pre-Trained Weights
The pre-trained model weights of NC-Net for ```CoNSeP```, ```PanNuke``` and ```Lizard``` can be found [here]([https://drive.google.com/drive/u/4/folders/1bndhWtwgsQLrNvupEpZqG9mEDWaWx89t](https://nustedupk0-my.sharepoint.com/:f:/g/personal/srashid_dphd19seecs_student_nust_edu_pk/Einy-z7qkNdJiHhdK7MvSi4BmgkeQ8Mt9G4pmwIaEnceVA?e=RWBiZe)).

### Model Inference
The model inference can done by running the command below:
```console
python inference.py
```
> **Note**: The training job currently only works with single GPU and does not support CPU training.

### Model Evaluation
For evaluation we are using using ```Dice Score```, ```Aggergarted Jaccard Index (AJI)```, ```Panoptic Quality (SQ)```, ```Detection (DQ)``` and ```Panoptic Quality (PQ)```. The model evaluation can be done using the command below:
```console
python verify.py
```
> **Note**: The training job currently only works with single GPU and does not support CPU training.
