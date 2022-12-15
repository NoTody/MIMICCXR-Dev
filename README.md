# MIMICXR-MutliModal-SelfSupervision
Multi-Modal Self-Supervision Pre-training BenchMarking on training with MIMICCXR and evaluating with CheXpert

Evaluation Method:

![alt text](https://github.com/NoTody/MIMICCXR-MutliModal-SelfSupervision/blob/main/imgs/architecture_example.png?raw=true)

Example pretraining command:
```
python ../main_pretrain.py --batch_size 32 --gpus 4 --num_nodes 1 --max_epochs 25 --lr_backbone 1e-4 --lr_projector 1e-4 --im    g_backbone "vit2d_b16" --max_length 128 --features_dim 768 --img_embedding_dim 768 --weight_decay 0.1 --optimizer "adamw" --method "SLIP_SIMCLR" --save_dir "slip_saved" --two_transform --pretrained --seed 2022
```

or

```
python main_pretrain.py --batch_size <batch_size> --gpus <num_gpu> --num_nodes <num_node> --max_epochs <num_epochs> --lr_backbone <backbone_learning_rate> --lr_projector <projector_learning_rate> --img_backbone <image_backbone_name> --max_length <text_tokenizer_length> --features_dim <feature_dimension> --img_embedding_dim <image_ebmedding_dimension> --optimizer <optimizer_name> --weight_decay <weight_decay> --method <train_method> --save_dir <save_directory> --two_transform --pretrained --seed 2022
```

Example finetuning command:
```
python ./chexpert_evaluation/base3.py --model_load_path <saved_model_path> --model_name "resnet50" --batch_size 64 --max_epoch 3 --save_suffix "resnet_slip_simclr" --seed 2022
```

or

```
python ./chexpert_evaluation/base3.py --model_load_path <saved_model_path> --model_name <model_name> --batch_size <batch_size> --max_epoch <num_epochs> --save_suffix <saved_model_suffix> --seed 2022

```

Examples on GradCam:

![alt text](https://github.com/NoTody/MIMICCXR-MutliModal-SelfSupervision/blob/main/imgs/grad_cam_example.png?raw=true)

