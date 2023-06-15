import argparse
import wandb
import torch
import numpy as np
import torch.nn as nn
from data.data_entry import select_train_valid_loader_full_batch, select_test_loader_full_batch
from utils import setup_seed
from options import parse_linear_reg_train_args
from tqdm import tqdm


import scipy.special as sc
import scipy

import json


# from opacus.accountants.utils import get_noise_multiplier
# from opacus.optimizers.optimizer import _generate_noise
# from torch.autograd.functional import jacobian
# from opt_einsum.contract import contract
# from opacus.accountants import create_accountant


def test(weights, test_loader, args):
    criterion = nn.MSELoss()
    losses = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            loss = criterion((weights @ inputs.T).view(-1), labels)
            losses.append(loss.item())
    return np.mean(losses)


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats as st
import math
import random


# Applies the PrivUnitG randomizer over input vector x
# x: input vector with unit norn
# eps: privacy parameter, if eps=-1 no privacy
# p,gamma,sigma: parameters for privG 
# n_trials: number of trials for sampling for the tail of gaussian
def PrivUnitG(x, eps, p=None, gamma=None, sigma=None, n_tries=None):
    if not p:
        p = priv_unit_G_get_p(eps)
    if p is None or gamma is None or sigma is None:
        gamma, sigma = get_gamma_sigma(p, eps)

    dim = x.size
    g = np.random.normal(0, 1, size = dim)
    pos_cor = np.random.binomial(1, p)

    if pos_cor:
        chosen_dps = np.array([sample_from_G_tail_stable(gamma)])
    else:
        if n_tries is None:
          n_tries = 25 # here probability of success is 1/2
        dps = np.random.normal(0, 1, size=n_tries)
        chosen_dps = dps[dps<gamma]
    
    if chosen_dps.size == 0:
        print('failure')
        return g * sigma
    target_dp = chosen_dps[0]

    # target_dp seems to be alpha


   
    yperp = g - (g.dot(x)) * x
    ypar = target_dp * x
    return target_dp * sigma, sigma * (yperp + ypar)




def get_gamma_sigma(p, eps):
    # Want p(1-q)/q(1-p) = exp(eps)
    # I.e q^{-1} -1 = (1-q)/q = exp(eps) * (1-p)/p
    qinv = 1 + (math.exp(eps) * (1.0-p)/ p)
    q = 1.0 / qinv
    gamma = st.norm.isf(q)
    # Now the expected dot product is (1-p)*E[N(0,1)|<gamma] + pE[N(0,1)|>gamma]
    # These conditional expectations are given by pdf(gamma)/cdf(gamma) and pdf(gamma)/sf(gamma)
    unnorm_mu = st.norm.pdf(gamma) * (-(1.0-p)/st.norm.cdf(gamma) + p/st.norm.sf(gamma))
    sigma = 1./unnorm_mu
    return gamma, sigma


def priv_unit_G_get_p(eps, return_sigma=False):
    # Mechanism:
    # With probability p, sample a Gaussian conditioned on g.x \geq gamma
    # With probability (1-p), sample conditioned on g.x \leq gamma
    # Scale g appropriately to get the expectation right
    # Let q(gamma) = Pr[g.x \geq gamma] = Pr[N(0,1) \geq gamma] = st.norm.sf(gamma)
    # Then density for x above threshold = p(x)  * p/q(gamma)
    # And density for x below threhsold = p(x) * (1-p)/(1-q(gamma))
    # Thus for a p, gamma is determined by the privacy constraint.
    plist = np.arange(0.01, 1.0, 0.01)
    glist = []
    slist = []
    for p in plist:
        gamma, sigma = get_gamma_sigma(p, eps)
        # thus we have to scale this rv by sigma to get it to be unbiased
        # The variance proxy is then d sigma^2
        slist.append(sigma)
        glist.append(gamma)
    ii = np.argmin(slist)
    if return_sigma:
        return plist[ii], slist[ii]
    else:
        return plist[ii]



# More stable version. Works at least until 1000
def sample_from_G_tail_stable(gamma):
    # return sample_from_G_tail(gamma)
    logq = scipy.stats.norm.logsf(gamma)
    u = np.random.uniform(low=0, high=1)
    logu = np.log(u)
    logr = logq + logu # r is now uniform in (0,q)
    #print(q,r)
    return -sc.ndtri_exp(logr)
    

# def inc_B(x, a, b):

#     return sc.betainc(a, b, x) / sc.gamma(a + b) * sc.gamma(a) * sc.gamma(b)


# def random_sample_sphere(d):
#     u = torch.randn(d)
#     m = u.pow(2).sum().pow(0.5)

#     return u / m



# def PrivUnit(v, p, gamma):

#     d = len(v)
#     v = v / v.pow(2).sum().pow(0.5)

    

#     if np.random.rand() < p:

#         u = random_sample_sphere(d)
#         print(u.multiply(v).sum())

#         while u.multiply(v).sum() < gamma:

#             print("case 1")

#             u = random_sample_sphere(d)
#             print(u.multiply(v).sum())
    
#     else:


#         u = random_sample_sphere(d)
#         print(u.multiply(v).sum())

#         while u.multiply(v).sum() >= gamma:

#             print("case 2")

#             u = random_sample_sphere(d)
#             print(u.multiply(v).sum())
    
#     V = u

#     alpha = (d - 1) / 2
#     tau = (1 + gamma) / 2

#     m = ((1 - (gamma ** 2)) ** gamma) / ((2.0 ** (d - 2)) * (d - 1))
#     m = m * ((p / inc_B(1, alpha, alpha) / inc_B(tau, alpha, alpha)) + (1 / p) / inc_B(tau, alpha, alpha))

#     return V / m



            




def experiment(args):


    setup_seed(args.seed)
    device = torch.device(args.device)

    # get multiple dataloader
    public_train_loader, private_train_loader, val_loader = select_train_valid_loader_full_batch(
        args)
    test_loader = select_test_loader_full_batch(args)

    # Linear Models weights
    weights = torch.randn(args.linear_reg_p, device=args.device)

    if args.warm_start:
        public_x, public_y = next(iter(public_train_loader))
        public_y = public_y.reshape((len(public_y), 1))
        optimal_theta = torch.linalg.solve(
            public_x.T @ public_x, public_x.T) @ public_y
        optimal_theta = optimal_theta.reshape(1, len(optimal_theta))
        weights = optimal_theta
        warm_start_loss = test(weights, val_loader, args)
        print(
            f"Warm Start Loss using public data: {warm_start_loss.item() :.10f}")
        
        data = {
            "throw_away":{
                "test_loss":warm_start_loss.item()
            }
        }
        if args.optimizer == "throw_away":
            return 0, 0, warm_start_loss.item()
            
        print()
        data_json_string = json.dumps(data, indent=4)
        print(data_json_string)
        print()

    weights = weights.view(-1)
    # calculate noise level using Accountant
    print(args.epsilon)
    print(args.delta)
    print(args.private_epochs)



    # gamma_max = (np.exp(args.epsilon) - 1) / (np.exp(args.epsilon) + 1) * np.sqrt(np.pi / (2 * (args.linear_reg_p - 1)))


    public_x, public_y = next(iter(public_train_loader))
    # public_y = public_y.reshape((len(public_y), 1))

    private_x, private_y = next(iter(private_train_loader))
    # private_y = private_y.reshape((len(private_y), 1))

    all_data_x = torch.cat([public_x, private_x], dim=0)
    all_data_y = torch.cat([public_y, private_y], dim=0)

    # 1: private; 0: public
    split = torch.cat([torch.zeros(len(public_x)), torch.ones(len(private_x))])

    idx = torch.torch.randperm(len(all_data_x))

    if args.priv_unit_g_p == -1:
        p = priv_unit_G_get_p(args.epsilon)
        gamma, sigma = get_gamma_sigma(p=p, eps=args.epsilon)
    else:
        gamma, sigma = get_gamma_sigma(p=args.priv_unit_g_p, eps=args.epsilon)

    ######################## get pub scaling factor #####################

    j = 0
    # find the 1st private
    while split[j] == 0:
        j += 1

    grad = all_data_x[j] * (all_data_x[j].dot(weights) - all_data_y[j])

    # ------ adding noise ------
    grad_normalized = grad / grad.pow(2).sum().pow(0.5)

    if args.priv_unit_g_p == -1:
        _, g = PrivUnitG(grad_normalized.numpy(), eps=args.epsilon, p=p, gamma=gamma, sigma=sigma, n_tries=1000)
    else:
        _, g = PrivUnitG(grad_normalized.numpy(), eps=args.epsilon, p=args.priv_unit_g_p, gamma=gamma, sigma=sigma, n_tries=1000)

    grad_noisy = torch.from_numpy(g)

    norm_grad_noisy_first = grad_noisy.pow(2).sum().pow(0.5)

    ######################## get pub scaling factor #####################


    

    for i in tqdm(idx):
        
        x = all_data_x[i]
        y = all_data_y[i]

        # always adding noise
        if args.optimizer == "ldp":

        

            grad = x * (x.dot(weights) - y)
    
            # ------ adding noise ------
            tmp = grad.pow(2).sum().pow(0.5)
            grad_normalized = grad / grad.pow(2).sum().pow(0.5)

            if args.priv_unit_g_p == -1:
                scaling_factor, g = PrivUnitG(grad_normalized.numpy(), eps=args.epsilon, p=p, gamma=gamma, sigma=sigma, n_tries=1000)
            else:
                scaling_factor, g = PrivUnitG(grad_normalized.numpy(), eps=args.epsilon, p=args.priv_unit_g_p, gamma=gamma, sigma=sigma, n_tries=1000)

            grad = torch.from_numpy(g)

            # ------ adding noise done ------

            weights -= args.lr * grad

        elif args.optimizer == "semi_ldp":


            grad = x * (x.dot(weights) - y)

            
            
    
            # ------ adding noise ------
            grad_normalized = grad / grad.pow(2).sum().pow(0.5)

            if args.priv_unit_g_p == -1:
                _, g = PrivUnitG(grad_normalized.numpy(), eps=args.epsilon, p=p, gamma=gamma, sigma=sigma, n_tries=1000)
            else:
                _, g = PrivUnitG(grad_normalized.numpy(), eps=args.epsilon, p=args.priv_unit_g_p, gamma=gamma, sigma=sigma, n_tries=1000)

            grad_noisy = torch.from_numpy(g)

            # ------ adding noise done ------

            if split[i] == 1:
                grad = grad_noisy
            
            else:
                norm_grad = grad.pow(2).sum().pow(0.5)
                norm_grad_noisy = grad_noisy.pow(2).sum().pow(0.5)


                # print(norm_grad_noisy, "????????")

                grad = grad / norm_grad * norm_grad_noisy_first
                # grad = grad / norm_grad * args.pub_grad_scaler
                

            weights -= args.lr * grad

            # print(grad.norm(), split[i])

        else:
            raise NotImplementedError


    private_train_loss = test(weights, private_train_loader, args)
    validate_loss = test(weights, val_loader, args)
    test_loss = test(weights, test_loader, args)

    print(
        f"/private train loss:{private_train_loss :.10f}"
        f"/validate loss: {validate_loss :.10f}"
        f"/test loss: {test_loss :.10f} "
    )

    wandb.log({"Valid loss":  private_train_loss,
                "Private train loss": validate_loss,
                "Test loss": test_loss
                })
    

    data = {
        args.optimizer: {
            'valid_loss': validate_loss,
            'private_train_loss': private_train_loss,
            'test_loss': test_loss,
            # 'cpu_time': elapsed_time_1 + elapsed_time_2,
            'best_params': {
                'lr': args.lr,
                'semi_dp_beta': args.semi_dp_beta,
                'public_private_ratio': args.public_private_ratio,
                'linear_reg_p': args.linear_reg_p,
                'epsilon': args.epsilon,
                'priv_unit_g_p': args.priv_unit_g_p,
            }
        },
    }

    print()
    data_json_string = json.dumps(data, indent=4)
    print(data_json_string)
    return validate_loss, private_train_loss, test_loss



def main():
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser = parse_linear_reg_train_args(parser)
    args = parser.parse_args()
    if not args.use_wandb:
        wandb.init(project=args.project_name, config=args, mode="disabled")
    else:
        wandb.init(project=args.project_name, config=args)
    print(f"Running on {args.device}")
    experiment(args)


if __name__ == '__main__':
    main()
