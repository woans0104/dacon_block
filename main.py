import random
import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
from collections import defaultdict

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



def main(config):
    delete_merge_folder = config.generating.delete_merge_folder
    train_generate_num = config.generating.train_generate_num
    val_generate_num = config.generating.val_generate_num
    all_generate_num = train_generate_num+val_generate_num 
    save_merged_folder = config.generating.folder_name
    train_one_sample_num = config.generating.train_one_sample_num
    val_one_sample_num = config.generating.val_one_sample_num
    all_one_sample_num = train_one_sample_num + val_one_sample_num
    
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    seed_everything(config.train.seed) # Seed 고정
    
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    # Data Load
    # Train / Validation Split
    df = pd.read_csv(config.data_dir.block_train)
    merge_df = pd.DataFrame(columns=df.columns)
    if delete_merge_folder:
        # make merge data

        temp_df = df.copy()
        sums = np.sum(temp_df[labels], axis=1)
        temp_df['label_sum'] = sums

        for_merge_inds = []
        for label in labels:
            inds = temp_df[(temp_df['label_sum']==1)&(temp_df[label]==1)].index.tolist()[:all_one_sample_num]
            for_merge_inds += inds

        for_merge_df = temp_df.loc[for_merge_inds,:]
        df = df.drop(for_merge_inds, axis=0)

        merged_folder = f'./data/{save_merged_folder}'
        if os.path.isdir(merged_folder):
            shutil.rmtree(merged_folder)

        os.mkdir(merged_folder)
        
    else:
        pass

    df = df.sample(frac=1)
    train_len = int(len(df) * 0.8)
    train_data = df[:train_len]
    val_data = df[train_len:]

    ids = []
    img_paths = []
    temp_label_dict = {}
    for t_label in labels:
        temp_label_dict[t_label] = [0]*(all_generate_num)
    generate_obj = merge_images(for_merge_df)

    for i in tqdm(range(train_generate_num)):
        filename = f'MERGE_{i}'
        img_path = f'{merged_folder}/{filename}.jpg'
        k = np.random.randint(7,10)
        temp_labels = random.sample(labels, k)
        for temp_label in temp_labels:
            temp_label_dict[temp_label][i] = 1
        
        merge_image = generate_obj.make_new_data(target_labels=temp_labels,
                                                random_indices=[k for k in range(train_one_sample_num)])
        
        
        
        cv2.imwrite(img_path, merge_image)
        ids.append(f'{filename}')
        img_paths.append(f'{save_merged_folder}/{filename}.jpg')
    
    
    for i in tqdm(range(train_generate_num, all_generate_num)):
        filename = f'MERGE_{i}'
        img_path = f'{merged_folder}/{filename}.jpg'
        k = np.random.randint(7,10)
        temp_labels = random.sample(labels, k)
        for temp_label in temp_labels:
            temp_label_dict[temp_label][i] = 1
        merge_image = generate_obj.make_new_data(target_labels=temp_labels,
                                                random_indices=[k for k in range(train_one_sample_num, all_one_sample_num)])
        cv2.imwrite(img_path, merge_image)
        ids.append(f'{filename}')
        img_paths.append(f'{save_merged_folder}/{filename}.jpg')
    
    merge_df['id'] = ids
    merge_df['img_path'] = img_paths
    for t_label in labels:
        merge_df[t_label] = temp_label_dict[t_label]

    merge_df.index = [i for i in range(len(df), len(df)+(all_generate_num))]
    
    train_data = pd.concat([train_data, merge_df.iloc[:train_generate_num]], axis=0)
    val_data = pd.concat([val_data, merge_df.iloc[train_generate_num:all_generate_num]], axis=0)

    # Data Preprocessing

    train_labels = get_labels(train_data)
    val_labels = get_labels(val_data)



    train_transform = A.Compose([
                                A.Resize(config.train.data.img_size, config.train.data.img_size),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                                            max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    test_transform = A.Compose([
                                A.Resize(config.train.data.img_size, config.train.data.img_size),
                                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                                            max_pixel_value=255.0, always_apply=False, p=1.0),
                                ToTensorV2()
                                ])

    train_dataset = CustomDataset(train_data['img_path'].values, train_labels, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = config.train.batch_size, shuffle=True,
                              num_workers=config.train.num_workers)

    val_dataset = CustomDataset(val_data['img_path'].values, val_labels, test_transform)
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
