cd example

# compare adaptive and non-adaptive and sgd
# use 4 architectures: WideResNet, ResNet, VGG, PyramidNet

# seed 42
python train.py --rho 0.05 --wandb_name non_ada_wideres_seed_42  --nnmodel WideResNet
python train.py --adaptive --rho 2.0 --wandb_name ada_wideres_seed_42  --nnmodel WideResNet

python train.py --rho 0.05 --wandb_name non_ada_vgg_seed_42  --nnmodel VGG
python train.py --adaptive --rho 2.0 --wandb_name ada_vgg_seed_42  --nnmodel VGG

python train.py --rho 0.05 --wandb_name non_ada_pyramid_seed_42  --nnmodel PyramidNet
python train.py --adaptive --rho 2.0 --wandb_name ada_pyramid_seed_42  --nnmodel PyramidNet

python train.py --rho 0.05 --wandb_name non_ada_res_seed_42  --nnmodel ResNet
python train.py --adaptive --rho 2.0 --wandb_name ada_res_seed_42  --nnmodel ResNet

python3 train.py --wandb_name sgd_widres_42 --seed 42 --nnmodel WideResNet --optimizer SGD
python3 train.py --wandb_name sgd_vgg_42 --seed 42 --nnmodel VGG --optimizer SGD
python3 train.py --wandb_name sgd_pyramid_42 --seed 42 --nnmodel PyramidNet --optimizer SGD
python3 train.py --wandb_name sgd_res_42 --seed 42 --nnmodel ResNet --optimizer SGD

# seed 10
python train.py --rho 0.05 --wandb_name non_ada_wideres_seed_10  --nnmodel WideResNet --seed 10
python train.py --adaptive --rho 2.0 --wandb_name ada_wideres_seed_10  --nnmodel WideResNet --seed 10

python train.py --rho 0.05 --wandb_name non_ada_vgg_seed_10  --nnmodel VGG --seed 10
python train.py --adaptive --rho 2.0 --wandb_name ada_vgg_seed_10  --nnmodel VGG --seed 10

python train.py --rho 0.05 --wandb_name non_ada_pyramid_seed_10  --nnmodel PyramidNet --seed 10
python train.py --adaptive --rho 2.0 --wandb_name ada_pyramid_seed_10  --nnmodel PyramidNet --seed 10

python train.py --rho 0.05 --wandb_name non_ada_res_seed_10  --nnmodel ResNet --seed 10
python train.py --adaptive --rho 2.0 --wandb_name ada_res_seed_10  --nnmodel ResNet --seed 10

python3 train.py --wandb_name sgd_widres_10 --seed 10 --nnmodel WideResNet --optimizer SGD --seed 10
python3 train.py --wandb_name sgd_vgg_10 --seed 10 --nnmodel VGG --optimizer SGD --seed 10
python3 train.py --wandb_name sgd_pyramid_10 --seed 10 --nnmodel PyramidNet --optimizer SGD --seed 10
python3 train.py --wandb_name sgd_res_10 --seed 10 --nnmodel ResNet --optimizer SGD --seed 10

# seed 0
python train.py --rho 0.05 --wandb_name non_ada_wideres_seed_0  --nnmodel WideResNet --seed 0
python train.py --adaptive --rho 2.0 --wandb_name ada_wideres_seed_0  --nnmodel WideResNet --seed 0

python train.py --rho 0.05 --wandb_name non_ada_vgg_seed_0  --nnmodel VGG --seed 0
python train.py --adaptive --rho 2.0 --wandb_name ada_vgg_seed_0  --nnmodel VGG --seed 0

python train.py --rho 0.05 --wandb_name non_ada_pyramid_seed_0  --nnmodel PyramidNet --seed 0
python train.py --adaptive --rho 2.0 --wandb_name ada_pyramid_seed_0  --nnmodel PyramidNet --seed 0

python train.py --rho 0.05 --wandb_name non_ada_res_seed_0  --nnmodel ResNet --seed 0
python train.py --adaptive --rho 2.0 --wandb_name ada_res_seed_0  --nnmodel ResNet --seed 0

python3 train.py --wandb_name sgd_widres_0 --seed 0 --nnmodel WideResNet --optimizer SGD
python3 train.py --wandb_name sgd_vgg_0 --seed 0 --nnmodel VGG --optimizer SGD
python3 train.py --wandb_name sgd_pyramid_0 --seed 0 --nnmodel PyramidNet --optimizer SGD
python3 train.py --wandb_name sgd_res_0 --seed 0 --nnmodel ResNet --optimizer SGD