cd example
# varies rho
python train.py --rho 0.01 --wandb_name varies_rho_0.01
python train.py --rho 0.02 --wandb_name varies_rho_0.02
python train.py --rho 0.05 --wandb_name varies_rho_0.05
python train.py --rho 0.1 --wandb_name varies_rho_0.1
python train.py --rho 0.2 --wandb_name varies_rho_0.2
python train.py --rho 0.5 --wandb_name varies_rho_0.5

# compare adaptive and non-adaptive
python train.py --rho 0.05 --wandb_name non-ada
python train.py --adaptive --rho 2.0 --wandb_name ada

# different norm, L2 and L_infty
