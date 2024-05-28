# Covariate Dataset

## Enviroment Setting (Linux)
conda create -n leaf python=3.8 -y
conda activate leaf
pip install -r requirements.txt


## 1. FEMNIST
we use leaf dataset for generating covariate FEMNIST
First, we split the FEMNIST dataset of leaf using below code
```sh
cd data/leaf/data/femnist
./preprocess.sh -s niid --sf 1.0 -k 0 -t sample --tf 0.9 --smplseed 42 --spltseed 42
```


## 2. CelebA
To-do