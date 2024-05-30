import json
import os

import numpy as np
import wget
from ...ml.engine import ml_engine_adapter

cwd = os.getcwd()

import zipfile

from ...constants import FEDML_DATA_MNIST_URL
import logging


# def download_mnist(data_cache_dir):
#     if not os.path.exists(data_cache_dir):
#         os.makedirs(data_cache_dir, exist_ok=True)

#     file_path = os.path.join(data_cache_dir, "MNIST.zip")
#     logging.info(file_path)

#     # Download the file (if we haven't already)
#     if not os.path.exists(file_path):
#         wget.download(FEDML_DATA_MNIST_URL, out=file_path)

#     file_extracted_path = os.path.join(data_cache_dir, "MNIST")
#     if not os.path.exists(file_extracted_path):
#         with zipfile.ZipFile(file_path, "r") as zip_ref:
#             zip_ref.extractall(data_cache_dir)

def read_data(train_data_dir, test_data_dir, num_task, client_number):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of non-unique client` ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clients = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    train_files = train_files[:client_number]
    
    for f in train_files: # select one user data file
        
        user = f.split(".")[0]
        file_path = os.path.join(train_data_dir, user+'.json')
        temp_data = {}
        with open(file_path, "r") as inf:
            cdata = json.load(inf)
            # cdata {'task_id': {'x': [], 'y': []}, 'task_id': {'x': [], 'y': []} ...}
            
        clients.append(user)
        
        for task_id in range(num_task):
            temp_data[task_id] = cdata[f"task_{task_id}"]
            
        train_data[user] = temp_data
        test_file_path = os.path.join(test_data_dir, user+'.json')
    
        with open(test_file_path, "r") as inf:
            cdata = json.load(inf)
        temp_data = {}
        
        for task_id in range(num_task):
            temp_data[task_id] = cdata[f"task_{task_id}"]
            
        test_data[user] = temp_data

    # clients = sorted(cdata["users"])

    return clients, train_data, test_data


def get_task_batch(args, total_task_data, batch_size):
    num_task = args.num_task
    batched_task_data = {}
    
    for task_id in range(num_task):
        data = total_task_data[task_id]
        one_task_batch_data = batch_data(args, data, batch_size)
        batched_task_data[task_id] = one_task_batch_data
        
    return batched_task_data


def batch_data(args, data, batch_size):

    """
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    """
    data_x = np.array(data["x"]).reshape(-1, 1, 28, 28)
    data_y = np.array(data["y"])

    # randomly shuffle data
    np.random.seed(100)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    batch_data = list()
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i : i + batch_size]
        batched_y = data_y[i : i + batch_size]
        batched_x, batched_y = ml_engine_adapter.convert_numpy_to_ml_engine_data_format(args, batched_x, batched_y)
        batch_data.append((batched_x, batched_y))

    return batch_data


def load_partition_data_CV_MNIST(args, client_number, batch_size):
    train_path = os.path.join(args.data_cache_dir, args.dataset, "train")
    test_path = os.path.join(args.data_cache_dir, args.dataset, "test")
    users, train_data, test_data = read_data(train_path, test_path,
                                             args.num_task, client_number=client_number)
    train_data_num = dict()
    test_data_num = dict()

    train_data_local_dict = dict()
    test_data_local_dict = dict()
    train_data_local_num_dict = dict()
    train_data_global = dict()
    test_data_global = dict()
    client_idx = 0
    logging.info("loading data...")
    
    for user_idx, u in enumerate(users):
        user_train_data_num = {}
        user_test_data_num = {}
        
        for task_id in train_data[u].keys():
            user_train_data_num[task_id] = len(train_data[u][task_id]["y"])
            user_test_data_num[task_id] = len(test_data[u][task_id]["y"])
        
        
        train_data_local_num_dict[client_idx] = user_train_data_num

        # transform to batches
        train_batch = get_task_batch(args, train_data[u], batch_size)
        test_batch = get_task_batch(args, test_data[u], batch_size)

        # index using client index
        train_data_local_dict[client_idx] = train_batch
        test_data_local_dict[client_idx] = test_batch
        
        for task_id in range(args.num_task):
            if user_idx == 0:
                train_data_global[task_id] = train_batch[task_id]
                test_data_global[task_id] = test_batch[task_id]
            else:
                train_data_global[task_id] += train_batch[task_id]
                test_data_global[task_id] += test_batch[task_id]
                
        client_idx += 1
    
    for task_id in range(args.num_task):
        train_data_num[task_id] = len(train_data_global[task_id])
        test_data_num[task_id] = len(test_data_global[task_id])

    # train_data_global[task_id][batch_idx]
    logging.info("finished the loading data")
    client_num = client_idx
    class_num = 62

    return (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    )
