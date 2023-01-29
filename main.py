import random
import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
from collections import defaultdict
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

import argparse
import os
import shutil
import yaml
import wandb
from sklearn.model_selection import train_test_split

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from dataset import CustomDataset
from model import BaseModel
from config.load_config import load_yaml, DotDict
import warnings
warnings.filterwarnings(action='ignore')

from merge_data import merge_images

# Fixed RandomSeed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_labels(df):
    return df.iloc[:,2:].values



# train / validation

def train(config, model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.BCELoss().to(device)

    best_val_acc = 0
    best_model = None

    train_step = 0
    loss_value = 0
    for epoch in range(1,config.train.epoch + 1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, labels)
            loss_value += loss.item()

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            if train_step > 0 and train_step % 5 == 0 :
                mean_loss = loss_value / 5
                loss_value = 0

                print("training_step: {}, "
                      "training_loss: {:.5f}"
                      .format(train_step, mean_loss))

                if config.wandb_opt:
                    wandb.log({'train_step': train_step, 'trn_mean_loss': mean_loss})

            train_step += 1



        # validation
        _val_loss, _val_acc = validation(model, criterion, val_loader, device)
        if config.wandb_opt:
            wandb.log({'val_loss': mean_loss, 'val_acc': _val_acc})


        _train_loss = np.mean(train_loss)
        print(
            f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val ACC : [{_val_acc:.5f}]')



        if scheduler is not None:
            scheduler.step(_val_acc)

        # savel best model
        if best_val_acc < _val_acc:
            best_val_acc = _val_acc
            best_model = model

            print("Saving state, index:", epoch)
            save_param_dic = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_param_path = (
                    config.results_dir
                    + "/daconBlock_"
                    + repr(epoch)
                    + ".pth"
            )
            torch.save(save_param_dic, save_param_path)

    return best_model


def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    val_acc = []
    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            probs = model(imgs)

            loss = criterion(probs, labels)

            probs = probs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            preds = probs > 0.5
            batch_acc = (labels == preds).mean()

            val_acc.append(batch_acc)
            val_loss.append(loss.item())

        _val_loss = np.mean(val_loss)
        _val_acc = np.mean(val_acc)

    return _val_loss, _val_acc


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


def overlay_data(generate_num_list, merge_data_obj, labels, save_folder, 
                 auto_block_size=True, type_='VAL'):
    ids = []
    img_paths = []
    label_dict = {}
    generate_num = sum(generate_num_list)
    merge_df = pd.DataFrame()
    for t_label in labels:
        label_dict[t_label] = [0]*(generate_num)
    
    if not os.path.isdir(f'./data/{save_folder}'):
        os.mkdir(f'./data/{save_folder}')
    
    file_index = 0
    for t_k, t_generate_num in enumerate(generate_num_list):
        t_k=t_k+1
        for i in tqdm(range(t_generate_num)):
            filename = f'{type_}_MERGE_{file_index+1}'
            img_path = f'./data/{save_folder}/{filename}.jpg'
            
            temp_labels = random.sample(labels, t_k)

            for temp_label in temp_labels:
                label_dict[temp_label][file_index] = 1
            
            merge_image = merge_data_obj.make_new_data(target_labels=temp_labels,
                                                      auto_block_size=auto_block_size)

            cv2.imwrite(img_path, merge_image)
            ids.append(f'{filename}')
            img_paths.append(f'./{save_folder}/{filename}.jpg')
            file_index += 1

    merge_df['id'] = ids
    merge_df['img_path'] = img_paths

    for t_label in labels:
        merge_df[t_label] = label_dict[t_label]

    return merge_df




def main(config):
    
    save_merged_folder = config.generating.folder_name
    
    
    same_class_generate_option = config.generating.same_class_generate.option
    same_class_generate_ratio = config.generating.same_class_generate.ratio
    same_class_generate_over_num = config.generating.same_class_generate.over_num
    
    train_target_sample_num = config.generating.train_target_sample_num
    val_target_sample_num = config.generating.val_target_sample_num
    all_target_sample_num = train_target_sample_num + val_target_sample_num
    
    auto_block_size = config.generating.auto_block_size
    val_ratio = config.data.val_ratio
    using_existed_merged_exp = config.train.using_existed_merged_exp
    
    existed_folder = config.generating.existed_folder
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    seed_everything(config.train.seed) # Seed 고정
    
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    # Data Load
    # Train / Validation Split
    df = pd.read_csv(config.data_dir.block_train)
    sums = np.sum(df[labels],axis=1)
    df['sums'] = sums
    if not existed_folder:
        if using_existed_merged_exp==False or using_existed_merged_exp==None:
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


            train_merge_obj = merge_images(train_df, image_path='./data/train')
            train_merged_df = overlay_data(train_generate_num_list, train_merge_obj, 
                                             labels, save_merged_folder, 
                                           auto_block_size=auto_block_size, type_='TRAIN')


            val_merge_obj = merge_images(val_df, image_path='./data/train')
            val_merged_df = overlay_data(val_generate_num_list, val_merge_obj, 
                                             labels, save_merged_folder, 
                                         auto_block_size=auto_block_size, type_='VAL')

            # filtering rows
            train_inds = []
            for sums in np.unique(train_df['sums']):
                inds = train_df[train_df['sums']==sums].index.tolist()
                length = len(inds)
                if length>train_target_sample_num:
                    inds = random.sample(inds, train_target_sample_num)
                train_inds += inds
            train_df = train_df.loc[train_inds,:].reset_index(drop=True)

            val_inds = []
            for sums in np.unique(val_df['sums']):
                inds = val_df[val_df['sums']==sums].index.tolist()
                length = len(inds)
                if length>val_target_sample_num:
                    inds = random.sample(inds, val_target_sample_num)
                val_inds += inds
            val_inds = val_df.loc[val_inds,:].reset_index(drop=True)


            target_cols = train_merged_df.columns.tolist()
            train_data = pd.concat([train_df.loc[:,target_cols], train_merged_df], axis=0).reset_index(drop=True)
            val_data = pd.concat([val_df.loc[:,target_cols], val_merged_df], axis=0).reset_index(drop=True)

            train_data.to_csv(f'{config.results_dir}/train_data.csv', index=False)
            val_data.to_csv(f'{config.results_dir}/val_data.csv', index=False)





        else:
            train_data = pd.read_csv(f'./exp/{using_existed_merged_exp}/train_data.csv')
            val_data = pd.read_csv(f'./exp/{using_existed_merged_exp}/val_data.csv')

        # Data Preprocessing

        train_labels = get_labels(train_data)
        val_labels = get_labels(val_data)

    else:
        train_label_path = os.path.join(existed_folder, 'TRAIN', 'labels')
        val_label_path = os.path.join(existed_folder, 'VAL', 'labels')
        with open(os.path.join(train_label_path, 'train_list.pkl'), 'rb') as f:
            train_image_list = pickle.load(f)
        with open(os.path.join(val_label_path, 'val_list.pkl'), 'rb') as f:
            val_image_list = pickle.load(f)
        
        train_labels = np.load(os.path.join(train_label_path, 'train_labels.npy'))
        val_labels = np.load(os.path.join(val_label_path, 'val_labels.npy'))
        
    # train_transform : 여러가지 추가
    train_transform = A.Compose([
                                A.Resize(config.train.data.img_size, config.train.data.img_size),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                                            max_pixel_value=255.0, always_apply=False, p=1.0),
        
                                # train_transform : 여러가지 추가
                                A.OneOf([
                                    A.ColorJitter(p=1),
                                    A.GaussNoise(p=1),
                                    A.ToSepia(p=1),
                                    A.RandomBrightnessContrast(p=1),
                                    A.ChannelDropout(p=1)
                                ],p=.33),
                                A.OneOf([
                                    A.AdvancedBlur(p=1)
                                ], p=.33),

                                A.OneOf([
                                    A.Affine(p=1),
                                    A.ElasticTransform(p=1)
                                ], p=.33),
        
                                ToTensorV2()
                                ])

    test_transform = A.Compose([
                                A.Resize(config.train.data.img_size, config.train.data.img_size),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                                            max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])
    if not existed_folder:
        train_dataset = CustomDataset(train_data['img_path'].values, train_labels, transforms=train_transform)
        val_dataset = CustomDataset(val_data['img_path'].values, val_labels, transforms=test_transform)
    else:
        train_dataset = CustomDataset(train_image_list, train_labels, data_dir='', transforms=train_transform)
        val_dataset = CustomDataset(val_image_list, val_labels, data_dir='', transforms=test_transform)
        
    train_loader = DataLoader(train_dataset, batch_size = config.train.batch_size, shuffle=True,
                              num_workers=config.train.num_workers)

    
    val_loader = DataLoader(val_dataset, batch_size = config.train.batch_size, shuffle=False, num_workers=0)

    #run
    model = BaseModel()
    if (device.type == 'cuda') and (torch.cuda.device_count() >1) :
        print("multi gpu activate")
        #model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count()))).cuda()
        model = nn.DataParallel(model).cuda()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = config.train.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2,
                                                           threshold_mode='abs',min_lr=1e-8, verbose=True)

    infer_model = train(config, model, optimizer, train_loader, val_loader, scheduler, device)





if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="dacon_block")
    parser.add_argument("--yaml",
                        "--yaml_file_name",
                        default="",
                        type=str,
                        help="Load configuration")


    args = parser.parse_args()

    # load configure
    exp_name = args.yaml
    config = load_yaml(args.yaml)

    print("-" * 20 + " Options " + "-" * 20)
    print(yaml.dump(config))
    print("-" * 40)

    # Make result_dir
    res_dir = os.path.join("exp", args.yaml)
    config["results_dir"] = res_dir
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # Duplicate yaml file to result_dir
    shutil.copy(
        "config/" + args.yaml + ".yaml", os.path.join(res_dir, args.yaml) + ".yaml"
    )

    # wandb setting

    if config["wandb_opt"]:
        wandb.init(project="dacon_block", entity="pingu", name=exp_name)
        wandb.config.update(config)

    config = DotDict(config)


    # train start
    main(config)
