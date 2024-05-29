import os
import json
import random
import numpy as np
import pandas as pd



def split_data(DATABATH, SAVEPATH, dataset, num_task=10):
    """
    Split data for task
    """
    
    # Load data
    train_data_path = os.path.join(DATABATH, "train")
    test_data_path = os.path.join(DATABATH, "test")
    
    train_data_list = os.listdir(train_data_path)
    
    if not os.path.exists(os.path.join(SAVEPATH,'train')):
        os.makedirs(os.path.join(SAVEPATH,'train'))
    if not os.path.exists(os.path.join(SAVEPATH,'test')):
        os.makedirs(os.path.join(SAVEPATH,'test'))
    
    save_train_path = os.path.join(SAVEPATH, "train")
    save_test_path = os.path.join(SAVEPATH, "test")
    
    # filename: all_data_1_keep_0_train_9.json
    for i in range(len(train_data_list)):
        trainfile = train_data_list[i]
        testfile = trainfile.replace("train", "test")
        
        with open(os.path.join(train_data_path, trainfile), 'r') as f:
            train_data = json.load(f)
        
        with open(os.path.join(test_data_path, testfile), 'r') as f:
            test_data = json.load(f)            
        
        # Split data with num_task per clients
        idx_chunk = [i*10 for i in range(num_task+1)]
        
        for j in range(len(idx_chunk)-1): # Select clients same with num_task
            train_users = train_data['users'][idx_chunk[j]:idx_chunk[j+1]]
            first_user = train_users[0]
            temp_train = {}
            temp_test = {}
            print("------- Generate data for user {} -------".format(first_user))
            
            for task_idx, user in enumerate(train_users):
                tr_data = train_data['user_data'][user]
                ts_data = test_data['user_data'][user]
                
                temp_train[f'task_{task_idx}'] = tr_data
                temp_test[f'task_{task_idx}'] = ts_data
                print(f'task_{task_idx} in {first_user}: {user}')
                                
            # Save data
            with open(os.path.join(save_train_path, f'{first_user}.json'), 'w') as f:
                json.dump(temp_train, f)
            
            with open(os.path.join(save_test_path, f'{first_user}.json'), 'w') as f:
                json.dump(temp_test, f)

            print("------- Finish generating data for user {} -------".format(first_user))


if __name__ == "__main__":
    DATABATH = "./data/leaf/data/femnist/data/"
    dataset = 'femnist'

    SAVEPATH = f'./data/preproc/cv_emnist'
    num_task = 10
    
    split_data(DATABATH, SAVEPATH, dataset, num_task)