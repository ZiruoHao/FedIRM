# FedIRM
Our code is founded on [FedAvg](https://github.com/shaoxiongji/federated-learning.git), [ColorMNIST](https://github.com/mkmenta/ColorMNIST.git) and [IRM](https://github.com/facebookresearch/InvariantRiskMinimization.git).

This is code of the paper of 'CInvariant Federated Learning from Causal Learning Perspective'
Only experiments on IIR MNIST and CMNITS is produced by far.

Materials sorting is still in progress.
to be continued.

## Requirements
python>=3.6  
pytorch>=0.4

## Run

The MLP models are produced by:
> python [main_nn.py](main_nn.py)

Federated learning with MLP is produced by:
> python [main_fed.py](main_fed.py)

See the arguments in [options.py](utils/options.py). 

For example:
> python main_nn.py --dataset mnist --num_channels 3 --model mlp  --gpu 0 --lam 2 --iid --error 0.3 --epochs 200
> python main_fed.py --dataset mnist --num_channels 3 --model mlp  --gpu 0 --lam 0 --iid --error 0.3 --epochs 200

`lam` denotes the strength of invariant regularization. While lam=0 the model is ERM or FedAvg

`num_channels` must be 3.


## References
Shaoxiong Ji. (2018, March 30). A PyTorch Implementation of Federated Learning. Zenodo. http://doi.org/10.5281/zenodo.4321561
McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-Efficient Learning of Deep Networks from Decentralized Data. In Artificial Intelligence and Statistics (AISTATS), 2017.





