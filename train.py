import os
import argparse

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src_files.helper_functions.helper_functions import mAP, CutoutPIL, ModelEma, CustomDataset, add_weight_decay
from src_files.models import create_model
from src_files.loss_functions.losses import AsymmetricLoss
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
import yaml
import shutil
import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score

import albumentations as A
from albumentations.pytorch import transforms as a_transforms


parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
#parser.add_argument('--data', type=str, default='/home/MSCOCO_2014/')
#parser.add_argument('--lr', default=1e-4, type=float)
#parser.add_argument('--model-name', default='tresnet_l')
#parser.add_argument('--model-path', default='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ML_Decoder/tresnet_l_pretrain_ml_decoder.pth', type=str)
#parser.add_argument('--num-classes', default=80, type=int)

#parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
#parser.add_argument('--image-size', default=448, type=int, metavar='N', help='input image size (default: 448)')
#parser.add_argument('--batch-size', default=56, type=int, metavar='N', help='mini-batch size')

# ML-Decoder
#parser.add_argument('--use-ml-decoder', default=1, type=int)
#parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
#parser.add_argument('--decoder-embedding', default=768, type=int)
#parser.add_argument('--zsl', default=0, type=int)

def main():
    args = parser.parse_args()
    YAML_FILE = 'train.yaml'
    with open(f'./config/{YAML_FILE}') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    train_data_list_dir = config['data_dir']['train_data_list_dir']
    train_labels_path = config['data_dir']['train_labels_path']
    val_data_list_dir = config['data_dir']['val_data_list_dir']
    val_labels_path = config['data_dir']['val_labels_path']
    save_folder_name = config['save']['folder_name']
    epoch = config['train']['epoch']
    lr = config['train']['lr']
    cut_prob = config['train']['cut_prob']
    
    custom_aug_option = config['augmentation']['custom_option']
    
    fine_tuning = config['train']['fine_tuning']
    
    img_size = config['model']['img_size']
    batch_size = config['train']['batch_size']
    workers = config['train']['workers']
    
    result_path = os.path.join('./result', 'train', save_folder_name)
    Path(os.path.join(result_path, 'models')).mkdir(parents=True, exist_ok=True)
    shutil.copy(
        "config/" + YAML_FILE, os.path.join(result_path, YAML_FILE)
    )
    
    model_config = config['model']
    # Setup model
    print('creating model {}...'.format(model_config))
    #model = create_model(args).cuda()
    model = create_model(model_config).cuda()
    # local_rank = torch.distributed.get_rank()
    # torch.cuda.set_device(1)
    # model = torch.nn.DataParallel(model,device_ids=[0])
    aa = []
    if fine_tuning:
        for i, (name, param) in enumerate(model.named_parameters()):
            if 'embed_standart' in name or 'query_embed' in name:
                print(f'{name} is not freeze')
            else :
                param.requires_grad = False

    print('done')

    # COCO Data loading
    #instances_path_val = os.path.join(args.data, 'annotations/instances_val2014.json')
    #instances_path_train = os.path.join(args.data, 'annotations/instances_train2014.json')
    #data_path_val = args.data
    #data_path_train = args.data
    #data_path_val = f'{args.data}/val2014'  # args.data
    #data_path_train = f'{args.data}/train2014'  # args.data
    
    if custom_aug_option:
        train_transform = A.Compose([
                        
                        A.OneOf([
                            A.ColorJitter(p=1, brightness=(0.2,0.5), contrast=(0.2,0.5), saturation=(0.2,0.5), hue=(0.2,0.5)),
                            A.RandomBrightnessContrast(p=1, brightness_limit=0.5, contrast_limit=0.5),
                            A.HueSaturationValue(p=1, hue_shift_limit=40, sat_shift_limit=60, val_shift_limit=40),
                            A.Cutout(p=1, num_holes=16, max_h_size=16, max_w_size=16),
                            A.RandomCrop(p=1, height=320, width=320)
                        ],p=.33),
                        A.OneOf([
                            A.Rotate(p=1, limit=(-20,20)),
                            A.Flip(p=1)
                        ], p=.33),
                        A.OneOf([
                            A.Affine(p=1),
                            A.ElasticTransform(p=1)
                        ], p=.33),
                        A.Resize(img_size, img_size),
                        a_transforms.ToTensorV2()
                        ])
        
        val_transform = A.Compose([
                        A.Resize(img_size, img_size),
                        a_transforms.ToTensorV2()
                        ])
        
    else:
        train_transform = transforms.Compose([
                          transforms.Resize((img_size, img_size)),
                          CutoutPIL(cutout_factor=0.5),
                          RandAugment(),
                          transforms.ToTensor(),
                          # normalize,
                      ])
        
        val_transform = transforms.Compose([
                                    transforms.Resize((img_size, img_size)),
                                    transforms.ToTensor(),
                                    # normalize, # no need, toTensor does normalization
                                ])
        
    val_dataset = CustomDataset(val_data_list_dir,
                                val_labels_path,
                                val_transform, custom_transform=custom_aug_option
                                )
    
    train_dataset = CustomDataset(train_data_list_dir,
                                  train_labels_path,
                                  train_transform, custom_transform=custom_aug_option)
 
    print("len(val_dataset)): ", len(val_dataset))
    print("len(train_dataset)): ", len(train_dataset))

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=False)

    # Actuall Training
    train_multi_label_coco(model, train_loader, val_loader, lr, result_path, epoch, cut_prob)


def train_multi_label_coco(model, train_loader, val_loader, lr, save_path, Epochs, cut_prob):
    ema = ModelEma(model, 0.9997)  # 0.9997^641=0.82
    Sig = torch.nn.Sigmoid()
    # set optimizer
    #Epochs = 40
    weight_decay = 1e-4
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    parameters = add_weight_decay(model, weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.2)
    
    train_accs = []
    val_accs = []
    train_losses = []
    val_accs = []
    #train_maps = []
    val_maps = []
    highest_mAP = 0
    trainInfoList = []
    scaler = GradScaler()
    best_val_acc = 0
    for epoch in range(Epochs):
        acc_sum = 0
        length = 0
        train_loss_sum = 0
        ind = 1
        for i, (inputData, target, _) in enumerate(train_loader):
            if inputData.dtype!=torch.HalfTensor:
                inputData = inputData.type(torch.HalfTensor)
            inputData = inputData.cuda()
            target = target.cuda()
            
            #target = target.max(dim=1)[0]
            with autocast():  # mixed precision
                output = model(inputData).float()  # sigmoid will be done in loss !
                
            loss = criterion(output, target)
            model.zero_grad()
            
            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            scheduler.step()
            ema.update(model)
            
            output_regular = Sig(output)
            acc_preds = (output_regular>cut_prob).type(torch.int)
            acc_sum += torch.sum(torch.mean((target==acc_preds).type(torch.float), axis=1)).item()
            length += output_regular.shape[0]
            
            train_loss_sum+=loss.item()
            ind += 1
            # store information
            if i % 100 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], LR {:.1e}, Loss: {:.1f}'
                      .format(epoch, Epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              scheduler.get_last_lr()[0], \
                              loss.item()))
                
        try:
            torch.save(model.state_dict(), os.path.join(save_path,
                'models', 'model-{}-{}.ckpt'.format(epoch + 1, i + 1)))
        except:
            pass

        model.eval()

        mAP_score, val_acc = validate_multi(val_loader, model, ema, cut_prob)
        model.train()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            try:
                torch.save(model.state_dict(), os.path.join(save_path, 
                    'models', 'model-highest.ckpt'))
            except:
                pass
        
        train_acc = acc_sum/length
        train_loss = train_loss_sum/ind
        train_accs.append(train_acc)
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        val_maps.append(mAP_score)
        
        with open(os.path.join(save_path, 'train_acc.pkl'), 'wb') as f:
            pickle.dump(train_accs, f)
        with open(os.path.join(save_path, 'train_loss.pkl'), 'wb') as f:
            pickle.dump(train_losses, f)
        with open(os.path.join(save_path, 'val_acc.pkl'), 'wb') as f:
            pickle.dump(val_accs, f)
        with open(os.path.join(save_path, 'val_map.pkl'), 'wb') as f:
            pickle.dump(val_maps, f)
        print('current_mAP = {:.2f}, highest_mAP = {:.2f}'.format(mAP_score, highest_mAP))
        print('current_train accuracy = {:.2f}'.format(train_acc))
        print('current_val_accuracy = {:.2f}, highest_val_accuracy = {:.2f}\n'.format(val_acc, best_val_acc))
        

def validate_multi(val_loader, model, ema_model, cut_prob):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    acc_sum = 0
    length = 0
    for i, (input, target, _) in enumerate(val_loader):
        if input.dtype!=torch.HalfTensor:
            input = input.type(torch.HalfTensor)
        target = target

        # compute output
        with torch.no_grad():
            with autocast():
                output = model(input.cuda())
                output_regular = Sig(output).cpu()
                output_ema = Sig(ema_model.module(input.cuda())).cpu()
        

        # for mAP calculation
        preds_regular.append(output_regular.cpu().detach())
        preds_ema.append(output_ema.cpu().detach())
        targets.append(target.cpu().detach())

        acc_preds = (output_regular>cut_prob).type(torch.int)
        acc_sum += torch.sum(torch.mean((target==acc_preds).type(torch.float), axis=1)).item()
        length += output_regular.shape[0]

    acc = acc_sum/length
    mAP_score_regular = mAP(torch.cat(targets).numpy(), torch.cat(preds_regular).numpy())
    mAP_score_ema = mAP(torch.cat(targets).numpy(), torch.cat(preds_ema).numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    return max(mAP_score_regular, mAP_score_ema), acc


if __name__ == '__main__':
    main()
