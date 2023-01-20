# MIMICXR-MutliModal-SelfSupervision
Multi-Modal Self-Supervision Pre-training BenchMarking on training with MIMICCXR and evaluating with CheXpert

Evaluation Method:

![alt text](https://github.com/NoTody/MIMICCXR-MutliModal-SelfSupervision/blob/main/imgs/architecture_example.png?raw=true)

# **Example pretraining command:**
```
python ../main_pretrain.py --batch_size 32 --gpus 4 --num_nodes 1 --max_epochs 25 --lr_backbone 1e-4 --lr_projector 1e-4 --im    g_backbone "vit2d_b16" --max_length 128 --features_dim 768 --img_embedding_dim 768 --weight_decay 0.1 --optimizer "adamw" --method "SLIP_SIMCLR" --save_dir "slip_saved" --two_transform --pretrained --seed 2022
```

or

```
python main_pretrain.py --batch_size <batch_size> --gpus <num_gpu> --num_nodes <num_node> --max_epochs <num_epochs> --lr_backbone <backbone_learning_rate> --lr_projector <projector_learning_rate> --img_backbone <image_backbone_name> --max_length <text_tokenizer_length> --features_dim <feature_dimension> --img_embedding_dim <image_ebmedding_dimension> --optimizer <optimizer_name> --weight_decay <weight_decay> --method <train_method> --save_dir <save_directory> --two_transform --pretrained --seed 2022
```

# **Example finetuning command:**

## [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)

```
python base3.py --model_load_path <path_to_weights> --batch_size 64 --max_epoch 5 --save_suffix <suffix> --seed 5 --train_percent 0.01
```


## [NIH-ChestX-ray 14](https://nihcc.app.box.com/v/ChestXray-NIHCC)

```
python train_full.py --model_load_path <path_to_weights> --model_name "resnet50" --batch_size 16 --max_epoch 30 --save_suffix <suffix> --seed 5 --train_percent 0.1 --method "FT" --num_class 14
```


Examples on GradCam:

![alt text](https://github.com/NoTody/MIMICCXR-MutliModal-SelfSupervision/blob/main/imgs/grad_cam_example.png?raw=true)

