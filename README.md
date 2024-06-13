This repository is built on top of `https://github.com/davda54/sam`. To replicate the experiments, one need to install `requirements.txt` first, and register a wandb account to check the experiments (and change Line 59 in `example/train.py` as your API key).

- To replicate the varies radius rho experiment, run `varies_rho.sh`. 
- To replicate the adaptive SAM experiment, run `ada_nonada.sh`.
- To replicate the L^2 vs L^infty norm experiment, run `vary_norm.sh`.
- To replicate the normalization experiment, run `normalization_exp.py`.