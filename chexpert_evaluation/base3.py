import numpy as np
import pandas as pd
import time
import csv
import random
import argparse
import matplotlib.pyplot as plt
from PIL import Image
from barbar import Bar

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision.ops import sigmoid_focal_loss
from torch.utils.data import SubsetRandomSampler

from timm.models.vision_transformer import _create_vision_transformer
from timm.models.layers import PatchEmbed
from timm.models.vision_transformer import VisionTransformer, _cfg

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score

use_gpu = torch.cuda.is_available()
print(use_gpu)

train_path = '/gpfs/data/denizlab/Datasets/Public/CheXpert-v1.0/train.csv'
valid_path = '/gpfs/data/denizlab/Datasets/Public/CheXpert-v1.0/valid.csv'

Traindata = pd.read_csv(train_path)
Traindata = Traindata[Traindata['Path'].str.contains("frontal")] # use only frontal images
Traindata.to_csv('/gpfs/data/denizlab/Users/skr2369/Chexpert/CheXpert-v1/U1-V1/train_mod.csv', index = False)
print("Train data length:", len(Traindata))

Validdata = pd.read_csv(valid_path)
Validdata = Validdata[Validdata['Path'].str.contains("frontal")] # use only frontal images
Validdata.to_csv('/gpfs/data/denizlab/Users/skr2369/Chexpert/CheXpert-v1/U1-V1/valid_mod.csv', index = False)
print("Valid data length:", len(Validdata))

pathFileTrain = '/gpfs/data/denizlab/Users/skr2369/Chexpert/CheXpert-v1/U1-V1/train_mod.csv'
pathFileValid = '/gpfs/data/denizlab/Users/skr2369/Chexpert/CheXpert-v1/U1-V1/valid_mod.csv'

# Neural network parameters:
nnIsTrained = False     # pre-trained using ImageNet
nnClassCount = 14       # dimension of the output

# Training settings: batch size, maximum number of epochs
trBatchSize = 64 
trMaxEpoch = 3 

# Parameters related to image transforms: size of the down-scaled image, cropped image
imgtransResize = (320, 320)
imgtransCrop = 224

# Class names
class_names = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 
               'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 
               'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_load_path", type=str, default="None")
    parser.add_argument("--model_name", type=str, choices=["resnet50", "densenet121", "vitb16"] , default="resnet50")
    parser.add_argument("--save_suffix", type=str, default="")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--linear_prob", default=False, action='store_true')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epoch", type=int, default=3)
    parser.add_argument("--train_percent", type=float, default=0.01)
    args = parser.parse_args()
    return args


class CheXpertDataSet(Dataset):
    def __init__(self, data_PATH, transform = None, policy = "ones"):
        """
        data_PATH: path to the file containing images with corresponding labels.
        transform: optional transform to be applied on a sample.
        Upolicy: name the policy with regard to the uncertain labels.
        """
        image_names = []
        labels = []

        with open(data_PATH, "r") as f:
            csvReader = csv.reader(f)
            next(csvReader, None) # skip the header
            for line in csvReader:
                image_name = line[0]
                label = line[5:]
                
                for i in range(14):
                    if label[i]:
                        a = float(label[i])
                        if a == 1:
                            label[i] = 1
                        elif a == -1:
                            if policy == "ones":
                                label[i] = 1
                            elif policy == "zeroes":
                                label[i] = 0
                            else:
                                label[i] = 0
                        else:
                            label[i] = 0
                    else:
                        label[i] = 0
                
                image_names.append('/gpfs/data/denizlab/Datasets/Public/' + image_name)
        
#                 image_names.append('./' + image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        """Take the index of item and returns the image and its labels"""
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
    

class CheXpertTrainer():

    def train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, checkpoint, save_suffix):
        optimizer = optim.Adam(model.parameters(), lr = 0.0001, # setting optimizer & scheduler
                               betas = (0.9, 0.999), eps = 1e-08, weight_decay = 0) 
        loss = torch.nn.BCELoss() # setting loss function
        
        if checkpoint != None and use_gpu: # loading checkpoint
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            
        # Train the network
        lossMIN = 100000
        train_start = []
        train_end = []
        for epochID in range(0, trMaxEpoch):
            train_start.append(time.time()) # training starts
            losst = CheXpertTrainer.epochTrain(model, dataLoaderTrain, optimizer, trMaxEpoch, nnClassCount, loss)
            train_end.append(time.time()) # training ends
#             lossv = CheXpertTrainer.epochVal(model, dataLoaderVal, optimizer, trMaxEpoch, nnClassCount, loss)
            print("Training loss: {:.3f},".format(losst))#, "Valid loss: {:.3f}".format(lossv))
            
            if losst < lossMIN:
                lossMIN = losst
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 
                            'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 
                            'model_saved/' + 'm-epoch_FL_' + save_suffix + str(epochID + 1) + '.pth.tar')
                print('Epoch ' + str(epochID + 1) + ' [save] loss = ' + str(losst))
            else:
                print('Epoch ' + str(epochID + 1) + ' [----] loss = ' + str(losst))

        train_time = np.array(train_end) - np.array(train_start)
        print("Training time for each epoch: {} seconds".format(train_time.round(0)))
        params = model.state_dict()
        return params
       
        
    def epochTrain(model, dataLoaderTrain, optimizer, epochMax, classCount, loss):
        losstrain = 0
        model.train()

        for batchID, (varInput, target) in enumerate(Bar(dataLoaderTrain)):
            
            varTarget = target.cuda(non_blocking = True)
            varOutput = model(varInput)
            #lossvalue = sigmoid_focal_loss(varOutput, varTarget, alpha=0.25, gamma=2, reduction="mean")
            lossvalue = loss(varOutput, varTarget)
            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            
            losstrain += lossvalue.item()
         
        return losstrain / len(dataLoaderTrain)

    
    def computeAUROC(dataGT, dataPRED, classCount):
        # Computes area under ROC curve 
        # dataGT: ground truth data
        # dataPRED: predicted data
        outAUROC = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            try:
                outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            except ValueError:
                pass
        return outAUROC
    
    
    def test(model, dataLoaderTest, nnClassCount, checkpoint, class_names):
        cudnn.benchmark = True
        
        if checkpoint != None and use_gpu:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])

        if use_gpu:
            outGT = torch.FloatTensor().cuda()
            outPRED = torch.FloatTensor().cuda()
        else:
            outGT = torch.FloatTensor()
            outPRED = torch.FloatTensor()
       
        model.eval()
        
        with torch.no_grad():
            for i, (input, target) in enumerate(dataLoaderTest):

                target = target.cuda()
                outGT = torch.cat((outGT, target), 0).cuda()

                bs, c, h, w = input.size()
                varInput = input.view(-1, c, h, w)
            
                out = model(varInput)
                outPRED = torch.cat((outPRED, out), 0)
        aurocIndividual = CheXpertTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        print('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print(class_names[i], ' ', aurocIndividual[i])
        
        return outGT, outPRED
    
    
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
        #self.backbone.classifier = nn.Linear(in_features=self.feature_dim_in, out_features=features_dim, bias=True)
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
        #self.backbone.fc = nn.Linear(in_features=self.feature_dim_in, out_features=features_dim, bias=True)
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
     def __init__(self, size, features_dim, out_size, pretrained=False, freeze_pos_embed=False, **kwargs):
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


if __name__ == "__main__":
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

    IMAGENET_MEAN = [0.485, 0.456, 0.406]  # mean of ImageNet dataset(for normalization)
    IMAGENET_STD = [0.229, 0.224, 0.225]   # std of ImageNet dataset(for normalization)

    # Training settings: batch size, maximum number of epochs
    trBatchSize = args.batch_size 
    trMaxEpoch = args.max_epoch

    # Tranform data
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    transformList = []

    transformList.append(transforms.Resize((imgtransCrop, imgtransCrop))) # 224
    # transformList.append(transforms.RandomResizedCrop(imgtransCrop))
    # transformList.append(transforms.RandomHorizontalFlip())
    transformList.append(transforms.ToTensor())
    # transformList.append(normalize)
    transformSequence = transforms.Compose(transformList)

    # Load dataset
    datasetTrain = CheXpertDataSet(pathFileTrain, transformSequence, policy = "ones")
    print("Train data length:", len(datasetTrain))
    train_size = (int)(len(datasetTrain) * args.train_percent)
    train_idx = torch.randperm(len(datasetTrain))[:train_size]
    train_sampler = SubsetRandomSampler(train_idx)
    dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, 
                            shuffle=False,  num_workers=20, pin_memory=True,
                            sampler=train_sampler)

    datasetValid = CheXpertDataSet(pathFileValid, transformSequence)
    print("Valid data length:", len(datasetValid))
    dataLoaderVal = DataLoader(dataset=datasetValid, batch_size=trBatchSize, 
                            shuffle=False, num_workers=20, pin_memory=True)


    if args.model_name == "densenet121":
        size = 121 
        features_dim = 2048
        out_size = nnClassCount
        model = densenet_model(size, features_dim, out_size, pretrained=True).cuda() # Step 0: Initialize global model and load the model
    elif args.model_name == "resnet50":
        size = 50
        features_dim = 2048
        out_size = nnClassCount
        model = resnet_model(size, features_dim, out_size, pretrained=True).cuda() # Step 0: Initialize global model and load the model
    elif args.model_name == "vitb16":
        size = "base" 
        features_dim = 768
        out_size = nnClassCount
        model = vit_model(size, features_dim, out_size, pretrained=True).cuda() # Step 0: Initialize global model and load the model
    else:
        raise NotImplementedError("Model Not Implemented!")

    # check if freeze the backbone model
    if args.linear_prob:
        for param in model.backbone.parameters():
            param.requires_grad = False
    
    model = torch.nn.DataParallel(model).cuda()

    if args.model_load_path != "None":
        checkpoint = torch.load(args.model_load_path)
        state_dict = {k.replace("img_backbone.", "module."): v for k, v in checkpoint['state_dict'].items()}
        #print(state_dict.keys())
        load_status = model.load_state_dict(state_dict, strict=False)
        print(f"Load Status: {load_status}")

    #batch, losst, losse = 
    CheXpertTrainer.train(model, dataLoaderTrain, dataLoaderVal, nnClassCount, trMaxEpoch, checkpoint = None, save_suffix = args.save_suffix)
    print("Model trained")
    outGT, outPRED = CheXpertTrainer.test(model, dataLoaderVal, nnClassCount, None, class_names)
    print(f"outGT: {outGT}, outPRED: {outPRED}")


