# SAmQ

SAmQ is a toolkit to conduct state aggregation for dynamic discrete choice model estiamtion. 

## Main Funcion

The main function is in [`main.py`](https://github.com/gengsinong/SAmQ/blob/master/rl/ddm.py). It includes a list of hyperparameters.

## Tuning

The tunning is conducted by [`submit_wandb.sh`](https://github.com/gengsinong/SAmQ/blob/master/submit_wandb.sh). 
This file uses wandb. 


## Data Generation

To generate the decision-making data for dynamic discrete choice model estimation, we consider a synthetic setting similar to the bus engine application in the [Rust Method](https://www.google.com/search?q=rust+bus+engine&rlz=1C1CHBF_zh-TWUS812US812&oq=rust+bus+engine&aqs=chrome..69i57j0i22i30.2014j0j7&sourceid=chrome&ie=UTF-8). 
The specific setting is defined and detialed in [`env.bus_env`](https://github.com/gengsinong/SAmQ/blob/master/env/bus_env.py). 
Then, we conduct soft-q iteration to estiamte the optimal choice-specific value function, and generate data in [`rl.softQ`](https://github.com/gengsinong/SAmQ/blob/master/rl/softQ.py). 

## Important State Estiamtion

With the synthetic data, we fist estimate the important states using the [PQR method](https://arxiv.org/abs/2007.07443) in ['irl.pqr'](https://github.com/gengsinong/SAmQ/blob/master/irl/pqr.py). 
This module also conducts state aggregation. 

## Utility Estiamtion
Finally, the utility is estimated on the important states by [`rl.ddm`](https://github.com/gengsinong/SAmQ/blob/master/rl/ddm.py)
