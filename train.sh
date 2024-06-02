cd example
# varies rho
python train.py --adaptive false --rho 0.01
python train.py --adaptive false --rho 0.02
python train.py --adaptive false --rho 0.05
python train.py --adaptive false --rho 0.1
python train.py --adaptive false --rho 0.2
python train.py --adaptive false --rho 0.5

# compare adaptive and non-adaptive
python train.py --adaptive false
python train.py --adaptive true

# different norm, L2 and L_infty