# Optimal Differentially Private Learning with Public Data


This is the official repository for our paper 
[Optimal Differentially Private Learning with Public Data](https://arxiv.org/abs/2306.15056) by [Andrew Lowy](https://sites.google.com/view/andrewlowy/home), [Zeman Li](https://sites.google.com/usc.edu/zemanli/), [Tianjian Huang](https://tianjian-huang.net/), and [Meisam Razaviyayn](https://sites.usc.edu/razaviyayn/). 

## Abstract
Differential Privacy (DP) ensures that training a machine learning model does not leak private data. However, the cost of DP is lower model accuracy or higher sample complexity. In practice, we may have access to auxiliary public data that is free of privacy concerns. This has motivated the recent study of what role public data might play in improving the accuracy of DP models. In this work, we assume access to a given amount of public data and settle the following fundamental open questions: 

1. What is the optimal (worst-case) error of a DP model trained over a private data set while having access to side public data? What algorithms are optimal? 
2. How can we harness public data to improve DP model training in practice? We consider these questions in both the local and central models of DP. 

To answer the first question, we prove tight (up to constant factors) lower and upper bounds that characterize the optimal error rates of three fundamental problems: mean estimation, empirical risk minimization, and stochastic convex optimization. We prove that public data reduces the sample complexity of DP model training. Perhaps surprisingly, we show that the optimal error rates can be attained (up to constants) by either discarding private data and training a public model, or treating public data like it's private data and using an optimal DP algorithm. To address the second question, we develop novel algorithms which are "even more optimal" (i.e. better constants) than the asymptotically optimal approaches described above. For local DP mean estimation with public data, our algorithm is **optimal including constants**. Empirically, our algorithms show benefits over existing approaches for DP model training with side access to public data. 



## Environment Setup
This code is tested with Python 3.8 and PyTorch 2.0.0 with CUDA 11.8.

To install the required packages, run

```shell
git clone git@github.com:optimization-for-data-driven-science/DP-with-public-data.git
cd DP-with-public-data
pip install -r requirements.txt
```

## Usage

### Run linear regression experiment
```shell
./scripts/run_linear_experiment.sh
```

#### Important Parameters Options

`--model_type`: The model to train. (default: `linear_reg`)

`--data_type`: The dataset to use. (default: `linear_reg_exp_gaussian`)

`--warm_start`: Use "warm start" to initialize the weights of models. The weights of models is randomly initialized if the option is not included.

`--linear_reg_p`: Specify the dimension of linear regression problem. It should be a positive integer. 

`--public_private_ratio`: The ratio of public dataset to total dataset.

`--optimizer`: The differential private optimizer to use. (default: `semi_dp`) 

- `semi_dp`: The optimizer (**Algorithm 1**) proposed in our paper. If `semi_dp` is used, also need to specify `--semi_dp_beta`. 
- `pda_pdmd_linear`: The PDA-PDMD optimizer in their closed form for linear regression.
- `deep_mind`: The DP-SGD optimizer with re-parameterization of gradient clipping. 
- `throw_away_sgd`: The SGD optimizer which only trains on public data and throws away private data. 


`--semi_dp_beta`: Required if `semi_dp` is used in `--optimizer`. It should be a real number in `[0,1]`.

`--private_epochs`: Number of epochs of private dataset is trained.


### Run Cifar10 logistic regression experiment
```shell
./scripts/run_cifar10_experiment.sh
```
#### Important Parameters Options

`--model_type`: The model to train. (default: `cifar_10_cvx`)

`--data_type`: The dataset to use. (default: `cifar10`)

`--optimizer`: The differential private optimizer to use. Options: `[semi_dp, deep_mind, throw_away_sgd]` (default: `semi_dp`). 

`--semi_dp_beta`: Required if `semi_dp` is used in `--optimizer`. 

`--iterations`: Number of iterations of private dataset is trained.


### Run Cifar10 WideResNet-16-4 experiment
```shell
./scripts/run_cifar_cvx_iterations_nonconvex.sh
```
#### Important Parameters Options
Similiar to ["Run Cifar10 logistic regression experiment"](#run-cifar10-logistic-regression-experiment), except for one different parameters. 

`--model_type`: The model to train. (default: `cifar_10_wide_resnet`)

### Common Differential Privacy Options
`--epsilon` and `--delta`: (ϵ, δ) that is used in differential privacy.

`--max_grad_norm`: The clipping norm for DP-SGD (defualt: 1.0).


## Citation
If you find this repository helpful, please cite our paper:
```  
@article{lowy2023optimal,
  title={Optimal Differentially Private Learning with Public Data},
  author={Lowy, Andrew and Li, Zeman and Huang, Tianjian and Razaviyayn, Meisam},
  journal={arXiv preprint arXiv:2306.15056},
  year={2023}
}
```

## Licence
Please see [License](/License.txt)