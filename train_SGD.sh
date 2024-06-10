cd example
python3 train.py --wandb_name sgd_widres_42 --seed 42 --nnmodel WideResNet --optimizer SGD
python3 train.py --wandb_name sgd_vgg_42 --seed 42 --nnmodel VGG --optimizer SGD
python3 train.py --wandb_name sgd_pyramid_42 --seed 42 --nnmodel PyramidNet --optimizer SGD
python3 train.py --wandb_name sgd_res_42 --seed 42 --nnmodel ResNet --optimizer SGD
