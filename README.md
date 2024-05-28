# FCL-NAS

## Environment setup
create and activate conda environment named ```fcl_nas``` with ```python=3.9```
```sh
conda create -n fcl_nas python=3.9 -y
conda activate fcl_nas
```

Install Pytorch
For ```Linux``` with ```CUDA```
```sh
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
conda install -c conda-forge mpi4py openmpi
pip install -r requirements.txt
```

For ```Mac with Slicon Chip``` or ```other OS``` (CPU Version) 
```sh
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 cpuonly -c pytorch
conda install -c conda-forge mpi4py openmpi
pip install -r requirements.txts
```

## Dataset

### 1. COV_EMNIST (Covariate EMNIST)
The data consists of 10 task per clients (0~9)
- X data: data[f'task_{idx}']['x']
- y data: data[f'task_{idx}']['y']
