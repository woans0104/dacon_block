
import argparse
import argparse
import os
import wandb
from dataset import CustomDataset
from model import BaseModel
from config.load_config import load_yaml, DotDict

import pandas as pd
import os

import torch
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models

from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score
from utils import copyStateDict

def inference(model, test_loader, device):

    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)
            probs = model(imgs)
            probs = probs.cpu().detach().numpy()
            preds = probs > 0.5
            preds = preds.astype(int)
            predictions += preds.tolist()
    return predictions



def eval_main(config) :


    test = pd.read_csv(config.data_dir.block_test)

    test_transform = A.Compose([
        A.Resize(config.eval.data.img_size, config.eval.data.img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False,
                    p=1.0),
        ToTensorV2()
    ])

    test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config.eval.batch_size, shuffle=False, num_workers=0)

    model = BaseModel()
    net_param = torch.load(config.eval.trained_model)
    model.load_state_dict(copyStateDict(net_param["model"]))
    model = model.cuda()
    cudnn.benchmark = False

    model.eval()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    preds = inference(model, test_loader, device)



    # make save dir
    save_dir_name = config.eval.save_dir_name
    save_dir = os.path.join(os.path.join("exp", args.yaml), "{}".format(save_dir_name))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    #Submission
    submit = pd.read_csv(config.data_dir.block_submission)
    submit.iloc[:,1:] = preds
    submit.head()
    submit.to_csv(os.path.join(save_dir,'baseline_submit.csv'), index=False)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="dacon_block")
    parser.add_argument(
        "--yaml",
        "--yaml_file_name",
        default=" ",
        type=str,
        help="Load configuration",
    )
    args = parser.parse_args()

    # load configure
    config = load_yaml(args.yaml)
    config = DotDict(config)

    if config["wandb_opt"]:
        wandb.init(project="dacon_block", entity="pingu", name=args.yaml)
        wandb.config.update(config)



    eval_main(config)
