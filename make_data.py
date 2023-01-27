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

from randimage import get_random_image
from merge_data import merge_images, extract_img

import albumentations as A
transform = A.Compose([
                    A.OneOf([
                        A.ColorJitter(p=1, brightness=(0.2,0.5), contrast=(0.2,0.5), saturation=(0.2,0.5), hue=(0.2,0.5)),
                        A.RandomBrightnessContrast(p=1, brightness_limit=0.5, contrast_limit=0.5),
                        A.HueSaturationValue(p=1, hue_shift_limit=40, sat_shift_limit=60, val_shift_limit=40)
                    ],p=.33),
                    A.OneOf([
                        A.Rotate(p=1, limit=(-10,10)),
                        A.Flip(p=1)
                    ], p=.33),
                    A.OneOf([
                        A.Affine(p=1),
                        A.ElasticTransform(p=1)
                    ], p=.33)
                    ])

LABEL_ID = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8 ,'J':9}
YAML_FILE = 'make_data'

# Fixed RandomSeed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)



def get_labels(df):
    return df.iloc[:,2:].values


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


def make_background(background_save_path, background_num, img_size=(400,400)):

    for i in tqdm(range(background_num)):
        img = get_random_image(img_size)
        img_file_name = f'background{i+1}.jpg'
        img_save_path = f'{os.path.join(background_save_path, img_file_name)}'
        cv2.imwrite(img_save_path, img*255)
    
    return 0



def main(config):
    
    # seed 고정
    seed_everything(config.seed) 
    save_merged_folder = config.generating.folder_name

    train_background_option = config.background.train_option
    val_background_option = config.background.val_option
    
    # background folder 생성
    background_save_path = ''
    if train_background_option:
        background_save_path = os.path.join('./data', save_merged_folder, 'TRAIN', 'background')
        Path(background_save_path).mkdir(parents=True, exist_ok=True)

    val_background_save_path = ''
    if val_background_option:
        val_background_save_path = os.path.join('./data', save_merged_folder, 'VAL', 'background')
        Path(val_background_save_path).mkdir(parents=True, exist_ok=True)
    
    
    # data 생성 yaml 파일 저장
    shutil.copy(
        "config/" + YAML_FILE + ".yaml", os.path.join('./data', save_merged_folder, YAML_FILE) + ".yaml"
    )
    
    # parameter 초기화
    same_class_generate_option = config.generating.same_class_generate.option
    same_class_generate_ratio = config.generating.same_class_generate.ratio
    same_class_generate_over_num = config.generating.same_class_generate.over_num
    
    train_target_sample_num = config.generating.train_target_sample_num
    val_target_sample_num = config.generating.val_target_sample_num
    all_target_sample_num = train_target_sample_num + val_target_sample_num
    
    auto_block_size = config.generating.auto_block_size
    val_ratio = config.data.val_ratio
    train_ratio = 1-val_ratio
    
    split_method = config.sampling.split_method
    under_sampling_option = config.sampling.under_sampling
    
    make_test = config.make_test.make_test
    test_folder_name = config.make_test.test_folder_name
    
    
    background_num = config.background.num
    background_existed_path = config.background.existed_path
    val_background_existed_path = config.background.val_existed_path
    
    
    train_transform_option = config.augmentation.train_transform_option
    val_transform_option = config.augmentation.val_transform_option
    
    # background 사용시 background 데이터 생성함
    background_path_list = []
    val_background_path_list = []
    if train_background_option:
        if background_existed_path:
            origin_data_list = os.listdir(background_existed_path)
            origin_path = [f'{os.path.join(background_existed_path, file)}' for file in origin_data_list if 'ipynb' not in file]
            new_path = [f'{os.path.join(background_save_path, file)}' for file in origin_data_list]
            
            for i in range(len(origin_path)):
                shutil.copy(origin_path[i], new_path[i])
            new_path_data_list = os.listdir(background_save_path)
            background_path_list = [f'{os.path.join(background_save_path, file)}' for file in new_path_data_list]
            
            
        else:
            make_background(background_save_path, background_num)
            background_path_list = [f'{os.path.join(background_save_path, file)}' for file in os.listdir(background_save_path)]
        
    if val_background_option:
        if val_background_existed_path:
            
            origin_data_list = os.listdir(val_background_existed_path)
            origin_path = [f'{os.path.join(val_background_existed_path, file)}' for file in origin_data_list if 'ipynb' not in file]
            new_path = [f'{os.path.join(val_background_save_path, file)}' for file in origin_data_list]

            for i in range(len(origin_path)):
                shutil.copy(origin_path[i], new_path[i])
            new_path_data_list = os.listdir(val_background_save_path)
            val_background_path_list = [f'{os.path.join(val_background_save_path, file)}' for file in new_path_data_list]
        else:
            make_background(val_background_save_path, background_num)
            val_background_path_list = [f'{os.path.join(val_background_save_path, file)}' for file in os.listdir(val_background_save_path)]
            #val_background_path_list = background_path_list.copy()

    # Data 초기화
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    # Data Load
    # Train / Validation Split
    df = pd.read_csv(config.data_dir.block_train)
    sums = np.sum(df[labels],axis=1)
    df['sums'] = sums
    
    # class, 블록개수 기준으로 train, valid 나눔 : 별 효과 없어서 안 쓸 것임
    if split_method=='rigid_split':

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


    elif split_method=='uniform':
                    
        # 블록 개수 기준으로 train, val 나눔
        """
        process
        - train, val에 해당하는 데이터 나눔 : 가진 데이터 기준으로만 생성할 것임
        - generate (합성)
          - train_target_sample_num, val_target_sample_num 기준으로 블록개수 당 generate해야 할 개수 구함
            - block이 1개인 경우는 제외함
          - 데이터 generate
        - train, val에 대해 정한 개수 넘는 블록개수에 해당하는 데이터는 random sample로 개수 조절함
        - 원본 데이터도 합성한 데이터의 폴더에 저장함
          - 배경 지정한 경우 배경 씌워서 저장함.
        """
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
    
    else:
        df = df.sample(frac=1)
        train_len = int(len(df) * train_ratio)

        train_inds = df[:train_len].index.tolist()
        val_inds = df[train_len:].index.tolist()
        
    # 정한 기준으로 train, val 나눔
    train_df = df.loc[train_inds, :].reset_index(drop=True)
    val_df = df.loc[val_inds, :].reset_index(drop=True)
    
    
    # generate_list
    train_generate_num_list = []
    val_generate_num_list = []
    
    # 생성할 개수 구함
    if same_class_generate_option:
        train_generate_num_list = make_class_generate_num(train_df, 
                                                 ratio=same_class_generate_ratio, 
                                                 over_num=same_class_generate_over_num,
                                                     target_num=train_target_sample_num)
        val_generate_num_list = make_class_generate_num(val_df, 
                                                     ratio=same_class_generate_ratio, 
                                                     over_num=same_class_generate_over_num,
                                                       target_num=val_target_sample_num)
    
    # train 생성 객체 초기화 및 생성
    type_ = 'TRAIN'
    train_img_save_path = os.path.join('./data', save_merged_folder, type_, 'images')
    train_label_save_path = os.path.join('./data', save_merged_folder, type_, 'labels')
    train_merge_obj = merge_images(train_df, image_path='./data/train', background_path=background_path_list)
    train_img_paths, train_label_list = overlay_data(train_generate_num_list, train_merge_obj, 
                                                     labels, train_img_save_path, train_label_save_path,
                                                     auto_block_size=auto_block_size, type_=type_)
    
    # val 생성 객체 초기화
    type_ = 'VAL'
    val_img_save_path = os.path.join('./data', save_merged_folder, type_, 'images')
    val_label_save_path = os.path.join('./data', save_merged_folder, type_, 'labels')

    val_merge_obj = merge_images(val_df, image_path='./data/train', background_path=val_background_path_list)
    val_img_paths, val_label_list = overlay_data(val_generate_num_list, val_merge_obj, 
                                     labels, val_img_save_path, val_label_save_path, 
                                 auto_block_size=auto_block_size, type_=type_)
    
    
    # train_target_sample_num 넘는 경우 제외함
    if train_target_sample_num and under_sampling_option:
        train_inds = []
        for sums in np.unique(train_df['sums']):
            inds = train_df[train_df['sums']==sums].index.tolist()
            length = len(inds)
            if length>train_target_sample_num:
                inds = random.sample(inds, train_target_sample_num)
            train_inds += inds
        train_df = train_df.loc[train_inds,:].reset_index(drop=True)
    
    # train 원본 데이터를 생성한 폴더 안에 복사함
    train_id_list = train_df['id'].tolist()
    origin_train_image_path = [f'./data/train/{id_}.jpg' for id_ in train_id_list]
    add_train_img_paths = [f'{train_img_save_path}/{id_}.jpg' for id_ in train_id_list]
    for i in tqdm(range(len(origin_train_image_path))):
        origin_path = origin_train_image_path[i]
        new_path = add_train_img_paths[i]
        
        if not background_path_list:
            shutil.copy(origin_path, new_path)
            
        else:
            background_target = random.choice(background_path_list)
            back_ground = cv2.imread(background_target)
            back_ground = cv2.resize(back_ground, (400,400))
            new_bbox, bbox_mask = extract_img(origin_path)
            bbox_y_list, bbox_x_list = np.where(bbox_mask<255)
            
            for y, x in zip(bbox_y_list, bbox_x_list):
                back_ground[y, x] = new_bbox[y,x,:]
                
                
            if train_transform_option:
                back_ground = transform(image=back_ground)['image']
                
            cv2.imwrite(new_path, back_ground)
        
    add_train_label_list = train_df[labels].values.tolist()
    
    # val_target_sample_num 넘는 경우 제외함
    if val_target_sample_num and under_sampling_option:
        val_inds = []
        for sums in np.unique(val_df['sums']):
            inds = val_df[val_df['sums']==sums].index.tolist()
            length = len(inds)
            if length>val_target_sample_num:
                inds = random.sample(inds, val_target_sample_num)
            val_inds += inds
        #val_inds = val_df.loc[val_inds,:].reset_index(drop=True)
        val_df = val_df.loc[val_inds,:].reset_index(drop=True)
    
    # val 원본 데이터를 생성한 폴더 안에 복사함
    val_id_list = val_df['id'].tolist()
    origin_val_image_path = [f'./data/train/{id_}.jpg' for id_ in val_id_list]
    add_val_img_paths = [f'{val_img_save_path}/{id_}.jpg' for id_ in val_df['id'].tolist()]
    for i in tqdm(range(len(origin_val_image_path))):
        origin_path = origin_val_image_path[i]
        new_path = add_val_img_paths[i]
        if not val_background_path_list and not transform_option:
            shutil.copy(origin_path, new_path)
            
        else:

            background_target = random.choice(val_background_path_list)
            back_ground = cv2.imread(background_target)
            back_ground = cv2.resize(back_ground, (400,400))
            new_bbox, bbox_mask = extract_img(origin_path)
            bbox_y_list, bbox_x_list = np.where(bbox_mask<255)

            for y, x in zip(bbox_y_list, bbox_x_list):
                back_ground[y, x] = new_bbox[y,x,:]

            if val_transform_option:
                back_ground = transform(image=back_ground)['image']

            cv2.imwrite(new_path, back_ground)
            
    add_val_label_list = val_df[labels].values.tolist()
    
    # path 지정 및 label 정보 가진 annotation 저장
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

    # test 데이터도 위와 같은 형식의 annotation 형태로 저장함
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
