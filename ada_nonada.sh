cd example

# compare adaptive and non-adaptive
# We only use 1 seed (42) now
# 8 runs
python train.py --rho 0.05 --wandb_name non_ada_wideres_seed_42  --nnmodel WideResNet
python train.py --adaptive --rho 2.0 --wandb_name ada_wideres_seed_42  --nnmodel WideResNet

python train.py --rho 0.05 --wandb_name non_ada_vgg_seed_42  --nnmodel VGG
python train.py --adaptive --rho 2.0 --wandb_name ada_vgg_seed_42  --nnmodel VGG

python train.py --rho 0.05 --wandb_name non_ada_pyramid_seed_42  --nnmodel PyramidNet
python train.py --adaptive --rho 2.0 --wandb_name ada_pyramid_seed_42  --nnmodel PyramidNet

python train.py --rho 0.05 --wandb_name non_ada_res_seed_42  --nnmodel ResNet
python train.py --adaptive --rho 2.0 --wandb_name ada_res_seed_42  --nnmodel ResNet