import random
import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import pickle

import argparse
import os
import shutil
import yaml


from tqdm.auto import tqdm
from config.load_config import load_yaml, DotDict
import warnings
warnings.filterwarnings(action='ignore')

from merge_data import merge_images

LABEL_ID = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8 ,'J':9}
YAML_FILE = 'make_data'

# Fixed RandomSeed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)



def get_labels(df):
    return df.iloc[:,2:].values



# train / validation




def make_class_generate_num(df, ratio=1, over_num=6, target_num=0):
    """
    target_num값 설정되면 ratio보다 우선됨
    """
    sums = df['sums']
    class_values = sums.value_counts()
    max_num = np.max(class_values)
    stand_generate_num = int(max_num*ratio)
    
    generate_list = [0,0,0,0,0,0,0,0,0,0]
    
    sum_values = sums.value_counts()
    index_set = set(sum_values.index.tolist())
    
    for i in range(len(generate_list)):
        num = i+1
        if num<over_num:
            continue
        
        now_num = 0

        if num in class_values:  
            now_num = class_values[num]
        if num in index_set:
            if target_num>0:
                generate_value = max(0, target_num-now_num)
            else:
                generate_value = max(0, stand_generate_num-now_num)
        else:
            if target_num>0:
                generate_value = target_num
            else:
                generate_value = stand_generate_num
    
        generate_list[i] = generate_value

    return generate_list


def overlay_data(generate_num_list, merge_data_obj, labels, img_save_path, label_save_path, 
                 auto_block_size=True, type_='VAL'):
    ids = []
    img_paths = []
    label_list = []
    #label_dict = {}
    generate_num = sum(generate_num_list)
    merge_df = pd.DataFrame()
    
    label_num = len(labels)
    
    Path(img_save_path).mkdir(parents=True, exist_ok=True)
    Path(label_save_path).mkdir(parents=True, exist_ok=True)
    # train
    train_paths = []
    file_index = 0
    for t_k, t_generate_num in enumerate(generate_num_list):
        t_k=t_k+1
        for i in tqdm(range(t_generate_num)):
            filename = f'{type_}_MERGE_{file_index+1}'
            img_path = f'{img_save_path}/{filename}.jpg'
            
            temp_labels = random.sample(labels, t_k)
            
            temp_label_list = [0]*label_num
            for temp_label in temp_labels:
                label_ind = LABEL_ID[temp_label]
                temp_label_list[label_ind] = 1

            merge_image = merge_data_obj.make_new_data(target_labels=temp_labels,
                                                      auto_block_size=auto_block_size)
            
            cv2.imwrite(img_path, merge_image)
            ids.append(f'{filename}')
            label_list.append(temp_label_list)
            img_paths.append(img_path)
            file_index += 1


    return img_paths, label_list




def main(config):
    
    
    seed_everything(config.train.seed) # Seed 고정
    
    save_merged_folder = config.generating.folder_name
    Path(os.path.join('./data', save_merged_folder)).mkdir(parents=True, exist_ok=True)
    shutil.copy(
        "config/" + YAML_FILE + ".yaml", os.path.join('./data', save_merged_folder, YAML_FILE) + ".yaml"
    )
    
    same_class_generate_option = config.generating.same_class_generate.option
    same_class_generate_ratio = config.generating.same_class_generate.ratio
    same_class_generate_over_num = config.generating.same_class_generate.over_num
    
    train_target_sample_num = config.generating.train_target_sample_num
    val_target_sample_num = config.generating.val_target_sample_num
    all_target_sample_num = train_target_sample_num + val_target_sample_num
    
    auto_block_size = config.generating.auto_block_size
    val_ratio = config.data.val_ratio
    train_ratio = 1-val_ratio
    using_existed_merged_exp = config.train.using_existed_merged_exp
    
    rigid_split = config.split_method.rigid_split
    
    make_test = config.make_test.make_test
    test_folder_name = config.make_test.test_folder_name
    
    
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    # Data Load
    # Train / Validation Split
    df = pd.read_csv(config.data_dir.block_train)
    sums = np.sum(df[labels],axis=1)
    df['sums'] = sums

    if rigid_split:
        
        label_list = [[] for _ in range(df.shape[0])]
        for label in labels:
            temp_list = df[label].tolist()
            for i, t_label in enumerate(temp_list):
                label_list[i].append(str(t_label))

        sums = df['sums'].tolist()
        for i, t_sum in enumerate(sums):
            label_list[i].append(str(t_sum))
        label_list = [''.join(t_label) for t_label in label_list]
            
        split_dict = {}
        index_encode_dict = {}
        index_decode_dict = {}
        unique_num = 1
        past_label = label_list[0]

        for i, t_label in enumerate(label_list):
            origin_t_label = t_label
            if past_label!=t_label:
                unique_num += 1

            t_label = str(unique_num).zfill(5)+t_label

            t_sum = int(t_label[-1])
            if t_sum not in split_dict.keys():
                split_dict[t_sum] = []
            if t_label not in set(split_dict[t_sum]):
                split_dict[t_sum].append(t_label)

            if t_label not in index_encode_dict.keys():
                index_encode_dict[t_label] = (i, unique_num)
                index_decode_dict[unique_num] = t_label
                #unique_num += 1
            past_label = origin_t_label
        
        
        train_inds = []
        val_inds = []
        for t_sum in split_dict.keys():

            t_labels = split_dict[t_sum]
            t_labels_length = len(t_labels)

            train_num = int(t_labels_length*train_ratio)

            if train_num<3 or t_sum==1:
                target_train_labels = t_labels
                target_val_labels = []

            else:

                target_train_labels = random.sample(t_labels, train_num)
                t_labels = set(t_labels).difference(target_train_labels)
                target_val_labels = list(t_labels)
                

            for target_label in target_train_labels:
                start_index, unique_num = index_encode_dict[target_label]
                next_unique_num = unique_num+1
                if next_unique_num not in index_decode_dict.keys():
                    end_index = df.shape[0]
                else:
                    next_target_label = index_decode_dict[next_unique_num]
                    end_index, _ = index_encode_dict[next_target_label]

                target_inds = [i for i in range(start_index, end_index)]
                train_inds += target_inds


            for target_label in target_val_labels:
                start_index, unique_num = index_encode_dict[target_label]
                next_unique_num = unique_num+1
                if next_unique_num not in index_decode_dict.keys():
                    end_index = df.shape[0]
                else:
                    next_target_label = index_decode_dict[next_unique_num]
                    end_index, _ = index_encode_dict[next_target_label]

                target_inds = [i for i in range(start_index, end_index)]
                val_inds += target_inds

            
        
    else:
        # split train, val
        temp_sums_dict = {}
        sums_list = df['sums'].tolist()
        for i in range(len(sums_list)):
            sums = sums_list[i]
            if sums not in temp_sums_dict.keys():
                temp_sums_dict[sums] = []

            temp_sums_dict[sums].append(i)

        train_inds = []
        val_inds = []
        split_least_num = int(1/val_ratio)
        for _, values in temp_sums_dict.items():
            length = len(values)
            if length<split_least_num:
                train_inds += values
            else:
                val_num = int(length*val_ratio)
                temp_val_inds = random.sample(values, val_num)
                temp_train_inds = list(set(values).difference(temp_val_inds))

                val_inds += temp_val_inds
                train_inds += temp_train_inds

    train_df = df.loc[train_inds, :].reset_index(drop=True)
    val_df = df.loc[val_inds, :].reset_index(drop=True)

    
    # generate_list
    train_generate_num_list = []
    val_generate_num_list = []
    if same_class_generate_option:
        train_generate_num_list = make_class_generate_num(train_df, 
                                                 ratio=same_class_generate_ratio, 
                                                 over_num=same_class_generate_over_num,
                                                     target_num=train_target_sample_num)
        val_generate_num_list = make_class_generate_num(val_df, 
                                                     ratio=same_class_generate_ratio, 
                                                     over_num=same_class_generate_over_num,
                                                       target_num=val_target_sample_num)

    type_ = 'TRAIN'
    train_img_save_path = os.path.join('./data', save_merged_folder, type_, 'images')
    train_label_save_path = os.path.join('./data', save_merged_folder, type_, 'labels')
    train_merge_obj = merge_images(train_df, image_path='./data/train')
    train_img_paths, train_label_list = overlay_data(train_generate_num_list, train_merge_obj, 
                                                     labels, train_img_save_path, train_label_save_path,
                                                     auto_block_size=auto_block_size, type_=type_)

    type_ = 'VAL'
    val_img_save_path = os.path.join('./data', save_merged_folder, type_, 'images')
    val_label_save_path = os.path.join('./data', save_merged_folder, type_, 'labels')
    

    val_merge_obj = merge_images(val_df, image_path='./data/train')
    val_img_paths, val_label_list = overlay_data(val_generate_num_list, val_merge_obj, 
                                     labels, val_img_save_path, val_label_save_path, 
                                 auto_block_size=auto_block_size, type_=type_)

    
    # filtering rows
    if train_target_sample_num:
        train_inds = []
        for sums in np.unique(train_df['sums']):
            inds = train_df[train_df['sums']==sums].index.tolist()
            length = len(inds)
            if length>train_target_sample_num:
                inds = random.sample(inds, train_target_sample_num)
            train_inds += inds
        train_df = train_df.loc[train_inds,:].reset_index(drop=True)
    
    train_id_list = train_df['id'].tolist()
    origin_train_image_path = [f'./data/train/{id_}.jpg' for id_ in train_id_list]
    add_train_img_paths = [f'{train_img_save_path}/{id_}.jpg' for id_ in train_id_list]
    for i in range(len(origin_train_image_path)):
        origin_path = origin_train_image_path[i]
        new_path = add_train_img_paths[i]
        shutil.copy(origin_path, new_path)
    add_train_label_list = train_df[labels].values.tolist()
    
    
    if val_target_sample_num:
        val_inds = []
        for sums in np.unique(val_df['sums']):
            inds = val_df[val_df['sums']==sums].index.tolist()
            length = len(inds)
            if length>val_target_sample_num:
                inds = random.sample(inds, val_target_sample_num)
            val_inds += inds
        #val_inds = val_df.loc[val_inds,:].reset_index(drop=True)
        val_df = val_df.loc[val_inds,:].reset_index(drop=True)
    
    val_id_list = val_df['id'].tolist()
    origin_val_image_path = [f'./data/train/{id_}.jpg' for id_ in val_id_list]
    add_val_img_paths = [f'{val_img_save_path}/{id_}.jpg' for id_ in val_df['id'].tolist()]
    for i in range(len(origin_val_image_path)):
        origin_path = origin_val_image_path[i]
        new_path = add_val_img_paths[i]
        shutil.copy(origin_path, new_path)
    add_val_label_list = val_df[labels].values.tolist()

    
    train_img_paths = train_img_paths+add_train_img_paths
    val_img_paths = val_img_paths+add_val_img_paths
    train_label_list = train_label_list+add_train_label_list
    val_label_list = val_label_list+add_val_label_list

    
    with open(f'{train_label_save_path}/train_list.pkl', 'wb') as f:
        pickle.dump(train_img_paths, f)
    with open(f'{val_label_save_path}/val_list.pkl', 'wb') as f:
        pickle.dump(val_img_paths, f)
    
    np.save(f'{train_label_save_path}/train_labels.npy', train_label_list)
    np.save(f'{val_label_save_path}/val_labels.npy', val_label_list)

    
    if make_test:
        test_df = pd.read_csv(config.data_dir.block_test)
        type_ = 'TEST'
        test_img_save_path = os.path.join('./data', test_folder_name, type_, 'images')
        test_label_save_path = os.path.join('./data', test_folder_name, type_, 'labels')
        
        Path(test_img_save_path).mkdir(parents=True, exist_ok=True)
        Path(test_label_save_path).mkdir(parents=True, exist_ok=True)
        
        test_id_list = test_df['id'].tolist()
        origin_test_image_path = [f'./data/test/{id_}.jpg' for id_ in test_id_list]
        add_test_img_paths = [f'{test_img_save_path}/{id_}.jpg' for id_ in test_id_list]

        for i in range(len(origin_test_image_path)):
            origin_path = origin_test_image_path[i]
            new_path = add_test_img_paths[i]
            shutil.copy(origin_path, new_path)
        
        test_img_paths = add_test_img_paths
        

        with open(f'{test_label_save_path}/test_list.pkl', 'wb') as f:
            pickle.dump(test_img_paths, f)
        

        
if __name__ == '__main__':


    
    config = load_yaml(YAML_FILE)
    
    print("-" * 20 + " Options " + "-" * 20)
    print(yaml.dump(config))
    print("-" * 40)


    # Duplicate yaml file to result_dir
    config = DotDict(config)
    
    # train start
    main(config)
