cd example
# varies  rho: 0.01, 0.05, 0.1, 0.5

# seed 0
python train.py --rho 0.01 --wandb_name varies_rho_0.01_seed_0  --seed 0
python train.py --rho 0.05 --wandb_name varies_rho_0.05_seed_0 --seed 0
python train.py --rho 0.1 --wandb_name varies_rho_0.1_seed_0 --seed 0
python train.py --rho 0.5 --wandb_name varies_rho_0.5_seed_0 --seed 0

# seed 10
python train.py --rho 0.01 --wandb_name varies_rho_0.01_seed_10  --seed 10
python train.py --rho 0.05 --wandb_name varies_rho_0.05_seed_10 --seed 10
python train.py --rho 0.1 --wandb_name varies_rho_0.1_seed_10 --seed 10
python train.py --rho 0.5 --wandb_name varies_rho_0.5_seed_10 --seed 10

# seed 42
python train.py --rho 0.01 --wandb_name varies_rho_0.01_seed_42  --seed 42
python train.py --rho 0.05 --wandb_name varies_rho_0.05_seed_42 --seed 42
python train.py --rho 0.1 --wandb_name varies_rho_0.1_seed_42 --seed 42
python train.py --rho 0.5 --wandb_name varies_rho_0.5_seed_42 --seed 42
