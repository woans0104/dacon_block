import os
import argparse
import time

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed

import albumentations as A
from albumentations.pytorch import transforms as a_transforms

import pandas as pd

from src_files.helper_functions.bn_fusion import fuse_bn_recursively
from src_files.models import create_model
from src_files.helper_functions.helper_functions import mAP, CutoutPIL, ModelEma, CustomDataset, add_weight_decay
import matplotlib
import torchvision.transforms as transforms

from src_files.models.tresnet.tresnet import InplacABN_to_ABN

matplotlib.use('TkAgg')
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import yaml
from pathlib import Path
import shutil


LABELS = ['A','B','C','D','E','F','G','H','I','J']
LABEL_ID_ENCODER = {key:i for i, key in enumerate(LABELS)}
LABEL_ID_DECODER = {value:key for key, value in LABEL_ID_ENCODER.items()}

parser = argparse.ArgumentParser(description='PyTorch MS_COCO infer')
#parser.add_argument('--num-classes', default=80, type=int)
#parser.add_argument('--model-path', type=str, default='./models_local/TRresNet_L_448_86.6.pth')
#parser.add_argument('--pic-path', type=str, default='./pics/000000000885.jpg')
#parser.add_argument('--model-name', type=str, default='tresnet_l')
#parser.add_argument('--image-size', type=int, default=448)
# parser.add_argument('--dataset-type', type=str, default='MS-COCO')
#parser.add_argument('--th', type=float, default=0.75)
#parser.add_argument('--top-k', type=float, default=20)
# ML-Decoder
#parser.add_argument('--use-ml-decoder', default=1, type=int)
#parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
#parser.add_argument('--decoder-embedding', default=768, type=int)
#parser.add_argument('--zsl', default=0, type=int)

def main():
    print('Inference code on a single image')
    
    # parsing args
    args = parser.parse_args()
    
    YAML_FILE = 'infer.yaml'
    with open(f'./config/{YAML_FILE}') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    
    model_path = config['infer']['model_path']
    
    
    batch_size = config['infer']['batch_size']
    workers = config['infer']['workers']
    top_k = config['infer']['top_k']
    th = config['infer']['th']
    
    save_folder_name = config['save']['folder_name']
    
    
    infer_data_list_dir = config['data_dir']['data_list_dir']
    infer_labels_path = config['data_dir']['labels_path']

    target_existed = False
    if infer_labels_path!=False and infer_labels_path!=None:
        target_existed = True
    
    train_folder = config['train']['folder_name']
    with open(f'./result/train/{train_folder}/train.yaml') as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    
    
    img_size = train_config['model']['img_size']
    model_name = train_config['model']['model_name']
    train_config['model']['model_path'] = model_path
    custom_aug_option = train_config['augmentation']['custom_option']
    
    result_path = os.path.join('./result', 'infer', save_folder_name)
    Path(result_path).mkdir(parents=True, exist_ok=True)
    shutil.copy(
        "config/" + YAML_FILE, os.path.join(result_path, YAML_FILE)
    )
    
    # Setup model
    print('creating model {}...'.format(model_name))
    model_config = train_config['model']

    model = create_model(model_config, load_head=True).cuda()
    state = torch.load(model_path, map_location='cpu')

    model.load_state_dict(state)
    ########### eliminate BN for faster inference ###########
    model = model.cpu()
    model = InplacABN_to_ABN(model)
    model = fuse_bn_recursively(model)
    model = model.cuda().half().eval()
    #######################################################
    print('done')

    
    
    
    if custom_aug_option:
        
        test_transform = A.Compose([
                        A.Resize(img_size, img_size),
                        a_transforms.ToTensorV2()
                        ])

    else:
        test_transform = transforms.Compose([
                          transforms.Resize((img_size, img_size)),
                          CutoutPIL(cutout_factor=0.5),
                          RandAugment(),
                          transforms.ToTensor(),
                          # normalize,
                      ])
    
    # doing inference
    infer_dataset = CustomDataset(infer_data_list_dir,
                                  infer_labels_path,
                                  test_transform, custom_transform=custom_aug_option
                                    # normalize, # no need, toTensor does normalization
                                )
    
    infer_loader = torch.utils.data.DataLoader(
        infer_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=False)

    
    obj = infer_multi(infer_loader, model, target_existed, th)
    
    result_df = pd.DataFrame()
    
    
    if target_existed:
        preds, targets, file_ids = infer_multi(infer_loader, model, target_existed, th)
        target_labels = [f'target_{t_label}' for t_label in LABELS]
        result_df[target_labels] = targets
        
    else:
        preds, file_ids = infer_multi(infer_loader, model, target_existed, th)
    result_df['id'] = file_ids
    result_df[LABELS] = preds
    result_df.to_csv(os.path.join(result_path, f'submission.csv'), index=False)
    
def infer_multi(val_loader, model, target_existed, th):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds = []

    targets = []
    file_ids = []
    for i, obj in enumerate(val_loader):
        if target_existed:
            input, target, file_name = obj
            target = target.cuda()
        else:
            input, file_name = obj
        if input.dtype!=torch.HalfTensor:
            input = input.type(torch.HalfTensor)
        input = input.cuda()
        # compute output
        with torch.no_grad():
            
            tensor_batch = torch.unsqueeze(input, 0).cuda()

            output = torch.squeeze(torch.sigmoid(model(input)))
            np_output = output.cpu().detach().numpy()
            binary_output = np_output > th
            binary_output = binary_output.astype(int)
            binary_output = [single_output.tolist() for single_output in binary_output]
            
            file_name = [name.split('/')[-1].split('.jpg')[0] for name in file_name]

            #idx_sort = np.argsort(-np_output)
            #detected_classes = np.array(classes_list)[idx_sort][: top_k]
            #scores = np_output[idx_sort][: top_k]
            #idx_th = np_output > th
            #detected_classes = detected_classes[idx_th]
            preds += binary_output
            file_ids += file_name
            if target_existed:
                targets.append(target.cpu().detach())
    if target_existed:
        return preds, targets, file_ids
    else:
        return preds, file_ids

    """
    print('loading image and doing inference...')
    im = Image.open(args.pic_path)
    im_resize = im.resize((img_size, img_size))
    np_img = np.array(im_resize, dtype=np.uint8)
    tensor_img = torch.from_numpy(np_img).permute(2, 0, 1).float() / 255.0  # HWC to CHW
    tensor_batch = torch.unsqueeze(tensor_img, 0).cuda().half() # float16 inference
    output = torch.squeeze(torch.sigmoid(model(tensor_batch)))
    np_output = output.cpu().detach().numpy()
    

    ## Top-k predictions
    # detected_classes = classes_list[np_output > args.th]
    idx_sort = np.argsort(-np_output)
    detected_classes = np.array(classes_list)[idx_sort][: top_k]
    scores = np_output[idx_sort][: top_k]
    idx_th = scores > th
    detected_classes = detected_classes[idx_th]
    print('done\n')

    # displaying image
    print('showing image on screen...')
    fig = plt.figure()
    plt.imshow(im)
    plt.axis('off')
    plt.axis('tight')
    # plt.rcParams["axes.titlesize"] = 10
    plt.title("detected classes: {}".format(detected_classes))

    plt.show()
    print('done\n')
    """

if __name__ == '__main__':
    main()
