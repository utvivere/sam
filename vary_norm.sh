# different norm, L2 and L_infty
# use seed 42
python train.py --wandb_name ptwo --rho 0.05
python train.py --wandb_name pinf --rho 0.05 --norm inf