# MIMICXR-MutliModal-SelfSupervision
Multi-Modal Self-Supervision Pre-training BenchMarking on MIMICCXR

Example command:
```
python main_pretrain.py --batch_size 64 --gpus 4 --num_nodes 1 --max_epochs 25 --lr_backbone 1e-4 --lr_projector 1e-4 --img_backbone "resnet2d_50" --max_length 128 --features_dim 768 --img_embedding_dim 2048 --optimizer "adamw" --weight_decay 0.1 --method "SLIP_MOCOV2" --save_dir "slip_mocov2_saved" --two_transform --pretrained
```
