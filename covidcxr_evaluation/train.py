# encoding: utf-8

"""
The main CheXNet model implementation.
"""
import sys
sys.path.append('..')
from read_data import CovidDataset
from eval_utils import *
from eval_utils import densenet_model, resnet_model, vit_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_load_path", type=str, default="None")
    parser.add_argument("--model_name", type=str, choices=["resnet50", "densenet121", "vitb16"] , default="resnet50")
    parser.add_argument("--save_suffix", type=str, default="")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--method", type=str , choices=["LP", "FT", "LPFT"] ,default="LPFT")
    parser.add_argument("--pretrained", default=False, action='store_true')
    parser.add_argument("--num_class", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epoch", type=int, default=5)
    parser.add_argument("--train_percent", type=float, default=1.0)
    args = parser.parse_args()
    return args


def compute_AUCs_covid(gt, pred):
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
    #AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    try:
        AUROC = roc_auc_score(gt_np, pred_np)
    except ValueError:
        pass
         
    return AUROC


def evaluate_covid(model, test_loader, N_CLASSES, CLASS_NAMES):
    pred = torch.FloatTensor()
    pred = pred.cuda()
    gt = torch.FloatTensor()
    gt = gt.cuda()
    model.eval()
    with torch.no_grad():
        for i, (inp, target) in enumerate(Bar(test_loader)):
            input_var, target = inp.cuda(), target.cuda()
            gt = torch.cat((gt, target), 0)
            output = model(input_var)
            pred = torch.cat((pred, output), 0)

    AUROC = compute_AUCs_covid(gt, pred)
    #AUROC_avg = np.array(AUROCs).mean()
    #print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    #for i in range(N_CLASSES):
    print('The AUROC of {} is {}'.format(CLASS_NAMES, AUROC))


def main(args):
    CLASS_NAMES = "COVID"
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.max_epoch
    N_CLASSES = args.num_class
    DATA_DIR = "/gpfs/data/denizlab/Users/hh2740/covid-cxr3/"
    DATA_DIR_TRAIN = f"{DATA_DIR}train"
    DATA_DIR_VAL = f"{DATA_DIR}train"
    DATA_DIR_TEST = f"{DATA_DIR}test"
    
    TRAIN_DF = f'{DATA_DIR}train_covid.txt'
    VAL_DF = f'{DATA_DIR}val_covid.txt'
    TEST_DF = f'{DATA_DIR}test_covid.txt'

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    transform_val = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    
    transform_test = transforms.Compose([
        #transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])

    train_dataset = CovidDataset(data_dir=DATA_DIR_TRAIN,
                                data_df_path=TRAIN_DF,
                                transform=transform_train)
   
    print("Train data length:", len(train_dataset))
    train_size = (int)(len(train_dataset) * args.train_percent)
    train_idx = torch.randperm(len(train_dataset))[:train_size]
    train_sampler = SubsetRandomSampler(train_idx) 

    val_dataset = CovidDataset(data_dir=DATA_DIR_VAL,
                            data_df_path=VAL_DF,
                            transform=transform_val)

    print("Valid data length:", len(val_dataset))
 
    test_dataset = CovidDataset(data_dir=DATA_DIR_TEST,
                                data_df_path=TEST_DF,
                                transform=transform_test)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
                             num_workers=10, pin_memory=True, sampler=train_sampler)

    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=10, pin_memory=True)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=10, pin_memory=True)

    if args.model_name == "densenet121":
        size = 121 
        features_dim = 2048
        out_size = N_CLASSES
        model = densenet_model(size, features_dim, out_size, pretrained=args.pretrained).cuda() # Step 0: Initialize global model and load the model
    elif args.model_name == "resnet50":
        size = 50
        features_dim = 2048
        out_size = N_CLASSES
        model = resnet_model(size, features_dim, out_size, pretrained=args.pretrained).cuda() # Step 0: Initialize global model and load the model
    elif args.model_name == "vitb16":
        size = "base" 
        features_dim = 768
        out_size = N_CLASSES
        model = vit_model(size, features_dim, out_size, pretrained=args.pretrained).cuda() # Step 0: Initialize global model and load the model
    else:
        raise NotImplementedError("Model Not Implemented!")

    model = torch.nn.DataParallel(model).cuda()
    criterion = torch.nn.BCELoss()

#     if args.model_load_path != "None":
#         checkpoint = torch.load(args.model_load_path)
#         state_dict = {k.replace("img_backbone.", "module."): v for k, v in checkpoint['state_dict'].items()}
#         load_status = model.load_state_dict(state_dict, strict=False)
#         print(f"Load Status: {load_status}")
    
    # freeze for linear probing
    if "LP" in args.method:
        if args.model_load_path != "None":
            checkpoint = torch.load(args.model_load_path)
            state_dict = {k.replace("img_backbone.", "module."): v for k, v in checkpoint['state_dict'].items()}
            load_status = model.load_state_dict(state_dict, strict=False)
            print(f"Load Status: {load_status}")
            
        for param in model.module.backbone.parameters():
            param.requires_grad = False

        # linear-probing
        print("---------------------Linear Probing---------------------")
        # initialize the ground truth and output tensor
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        best_model = train(args, model, optimizer, criterion, train_loader, val_loader, N_CLASSES, N_EPOCHS, CLASS_NAMES)

        # switch to evaluate mode
        print("Testing LP ...")
        evaluate_covid(best_model, test_loader, N_CLASSES, CLASS_NAMES)

    # fine-tuning
    if "FT" in args.method:
        print("---------------------Fine-Tuning---------------------")
        if args.method == "LPFT":
            model = best_model

        # unfreeze for finetuning
        for param in model.module.backbone.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999))
        best_model = train(args, model, optimizer, criterion, train_loader, val_loader, N_CLASSES, N_EPOCHS, CLASS_NAMES)

        # switch to evaluate mode
        print(f"Testing {args.method} ...")
        evaluate_covid(best_model, test_loader, N_CLASSES, CLASS_NAMES)


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

