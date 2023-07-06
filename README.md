# DP-with-public-data

This repo contains code to run experiments from our project on differential privacy with public data.

## Setup Environment

```shell
pip install -r requirements.txt
```

## Run linear regression experiment
```shell
./scripts/run_linear_experiment.sh
```
## Run Cifar10 logistic regression experiment
```shell
./scripts/run_cifar10_experiment.sh
```

## Run Cifar10 WideResNet-16-4 experiment
```shell
./scripts/run_cifar_cvx_iterations_nonconvex.sh
```

## Run LDP experiment
```shell
./scripts/run_linear_ldp_experiment.sh
```

