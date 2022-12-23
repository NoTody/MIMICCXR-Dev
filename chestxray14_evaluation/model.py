# encoding: utf-8

"""
The main CheXNet model implementation.
"""

import os
import argparse
import random
import numpy as np
import torch
import copy
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from read_data import ChestXrayDataSet
from sklearn.metrics import roc_auc_score
from barbar import Bar

from timm.models.vision_transformer import _create_vision_transformer
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer, _cfg

N_CLASSES = 14
CLASS_NAMES = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
DATA_DIR = '/gpfs/data/denizlab/Datasets/Public/NIH_Chest_X-ray/images'
TEST_IMAGE_LIST = 'test_list.txt'
TRAIN_LIST = 'train_list.txt'
VAL_LIST = 'val_list.txt'

def adjust_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 10

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_load_path", type=str, default="None")
    parser.add_argument("--model_name", type=str, choices=["resnet50", "densenet121", "vitb16"] , default="resnet50")
    parser.add_argument("--save_suffix", type=str, default="")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--linear_prob", default=False, action='store_true')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epoch", type=int, default=5)
    parser.add_argument("--train_percent", type=float, default=1.0)
    args = parser.parse_args()
    return args

def main(args):
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.max_epoch

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    train_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TRAIN_LIST,
                                    transform=transforms.Compose([
                                        transforms.RandomHorizontalFlip(),
                                        transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
   
    print("Train data length:", len(train_dataset))
    train_size = (int)(len(train_dataset) * args.train_percent)
    train_idx = torch.randperm(len(train_dataset))[:train_size]
    train_sampler = SubsetRandomSampler(train_idx) 

    val_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=VAL_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(224),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        normalize
                                    ]))
    print("Valid data length:", len(val_dataset))
 
    test_dataset = ChestXrayDataSet(data_dir=DATA_DIR,
                                    image_list_file=TEST_IMAGE_LIST,
                                    transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.TenCrop(224),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                                        transforms.Lambda
                                        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
                                    ]))
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                             num_workers=8, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=8, pin_memory=True)

    if args.model_name == "densenet121":
        size = 121 
        features_dim = 2048
        out_size = N_CLASSES
        model = densenet_model(size, features_dim, out_size, pretrained=True).cuda() # Step 0: Initialize global model and load the model
    elif args.model_name == "resnet50":
        size = 50
        features_dim = 2048
        out_size = N_CLASSES
        model = resnet_model(size, features_dim, out_size, pretrained=True).cuda() # Step 0: Initialize global model and load the model
    elif args.model_name == "vitb16":
        size = "base" 
        features_dim = 768
        out_size = N_CLASSES
        model = vit_model(size, features_dim, out_size, pretrained=True).cuda() # Step 0: Initialize global model and load the model
    else:
        raise NotImplementedError("Model Not Implemented!")

    # check if freeze the backbone model
    if args.linear_prob:
        for param in model.backbone.parameters():
            param.requires_grad = False

    model = torch.nn.DataParallel(model).cuda()
    best_model = copy.deepcopy(model)

    if args.model_load_path != "None":
        checkpoint = torch.load(args.model_load_path)
        state_dict = {k.replace("img_backbone.", "module."): v for k, v in checkpoint['state_dict'].items()}
        load_status = model.load_state_dict(state_dict, strict=False)
        print(f"Load Status: {load_status}")

    # initialize the ground truth and output tensor
    pred = torch.FloatTensor()
    pred = pred.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = torch.nn.BCELoss()
    best_loss = 1e5
    for epoch in range(N_EPOCHS):
        print(f"Training Epoch {epoch} ...")
        train_losses = 0
        model.train()
        for i, (inp, target) in enumerate(Bar(train_loader)):
            optimizer.zero_grad()
            inp, target = inp.cuda(), target.cuda()
            output = model(inp)
            train_loss = criterion(output, target)
            train_loss.backward()
            optimizer.step()
            train_losses += train_loss.item()
        
        train_losses /= len(train_loader)
        print("Training loss: {:.3f},".format(train_losses))

        print(f"Validating Epoch {epoch} ...")
        val_losses = 0
        model.eval()
        for i, (inp, target) in enumerate(Bar(val_loader)):
            inp, target = inp.cuda(), target.cuda()
            output = model(inp)
            val_loss = criterion(output, target)
            val_losses += val_loss.item()
           
        val_losses /= len(val_loader)
        print("Validation loss: {:.3f},".format(val_losses))
        if best_loss > val_losses:
            best_loss = val_losses
            best_model = model
            torch.save({'state_dict': model.state_dict(), 
                        'best_loss': best_loss, 'optimizer' : optimizer.state_dict()}, 
                        'model_saved/' + args.save_suffix + '.pth.tar')
            print('Epoch ' + str(epoch + 1) + ' [save] loss = ' + str(best_loss))
        else:
            print('Epoch ' + str(epoch + 1) + ' [----] loss = ' + str(best_loss))
            adjust_lr(optimizer)
        
    # switch to evaluate mode
    print("Testing ...")
    gt = torch.FloatTensor()
    gt = gt.cuda()
    model = best_model
    model.eval()
    with torch.no_grad():
        for i, (inp, target) in enumerate(Bar(test_loader)):
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            bs, n_crops, c, h, w = inp.size()
            input_var = torch.autograd.Variable(inp.view(-1, c, h, w).cuda())
            output = model(input_var)
            output_mean = output.view(bs, n_crops, -1).mean(1)
            pred = torch.cat((pred, output_mean.data), 0)

    AUROCs = compute_AUCs(gt, pred)
    AUROC_avg = np.array(AUROCs).mean()
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(N_CLASSES):
        print('The AUROC of {} is {}'.format(CLASS_NAMES[i], AUROCs[i]))


def compute_AUCs(gt, pred):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(N_CLASSES):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


class densenet_model(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, size, features_dim, out_size, pretrained=True):
        super(densenet_model, self).__init__()
        
        if size == 121:
            self.backbone = torchvision.models.densenet121(pretrained=pretrained)

        self.feature_dim_in = self.backbone.classifier.weight.shape[1]
        self.backbone.classifier = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim_in, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


# resnet model
class resnet_model(nn.Module):
    def __init__(self, size, features_dim, out_size, pretrained=True):
        super(resnet_model, self).__init__()
        
        if size==18:
            self.backbone = torchvision.models.resnet18(pretrained=pretrained)
        elif size==50:
            self.backbone = torchvision.models.resnet50(pretrained=pretrained)
        elif size==101:
            self.backbone = torchvision.models.resnet101(pretrained=pretrained)
        else:
            raise NotImplementedError(f"ResNet with size {size} is not implemented!")

        self.feature_dim_in = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim_in, out_size),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


# vit model
class vit_model(nn.Module):
     def __init__(self, size, features_dim, out_size, pretrained=True, freeze_pos_embed=False, **kwargs):
        super(vit_model, self).__init__()

        if freeze_pos_embed:
            pass
        else:
            if size=="base":
                model_kwargs = dict(
                    patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=0, **kwargs
                )
                self.backbone = _create_vision_transformer("vit_base_patch16_224", pretrained=pretrained, **model_kwargs)
            else:
                pass
        
        self.classifier = nn.Sequential(
                nn.Linear(features_dim, out_size),
                nn.Sigmoid()
        )

     def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    args = get_args()

    if args.seed != None:
        random_seed = args.seed
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

    main(args)


