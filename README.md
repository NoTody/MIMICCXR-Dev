# MIMICXR-MutliModal-SelfSupervision
Multi-Modal Self-Supervision Pre-training BenchMarking on training with MIMICCXR and evaluating with CheXpert

Example pretraining command:
```
python main_pretrain.py --batch_size 64 --gpus 4 --num_nodes 1 --max_epochs 25 --lr_backbone 1e-4 --lr_projector 1e-4 --img_backbone "resnet2d_50" --max_length 128 --features_dim 768 --img_embedding_dim 2048 --optimizer "adamw" --weight_decay 0.1 --method "SLIP_MOCOV2" --save_dir "slip_mocov2_saved" --two_transform --pretrained --seed 2022
```

Example finetuning command:
```
python base3.py --model_load_path <saved_model_path> --model_name "resnet50" --batch_size 64 --max_epoch 3 --save_suffix "resnet_slip_simclr" --seed 2022

```

Examples on GradCam:

![alt text](https://github.com/NoTody/MIMICCXR-MutliModal-SelfSupervision/blob/main/imgs/grad_cam_example.png?raw=true)
