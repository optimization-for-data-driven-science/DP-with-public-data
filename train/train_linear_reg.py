import argparse
import wandb
import torch
import json
import numpy as np
from opacus import PrivacyEngine
from torch import optim
import torch.nn as nn
from model.model_entry import select_model
from data.data_entry import select_train_valid_loader_full_batch, select_test_loader_full_batch
from optimizer.optimizer_entry import select_optimizer
from utils import setup_seed
from options import parse_linear_reg_train_args
import time


def test(model, test_loader, criterion, args):
    model.eval()
    losses = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), labels)
            losses.append(loss.item())
    model.train()
    return np.mean(losses)


def train_per_epoch(
        epoch,
        model,
        public_train_loader,
        private_train_loader,
        criterion,
        optimizer,
        args,
):
    device = torch.device(args.device)

    model.train()

    if args.optimizer in ['semi_dp', 'throw_away_sgd']:
        if public_train_loader is not None:
            data = next(iter(public_train_loader))
            optimizer.zero_grad()
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step_public()

    if args.optimizer in ['semi_dp', 'pda_pdmd_linear', 'deep_mind', 'dp_sgd']:
        if private_train_loader is not None:
            data = next(iter(private_train_loader))
            if args.optimizer == 'deep_mind' or args.optimizer == 'dp_sgd':
                public_data = next(iter(public_train_loader))
                data = [torch.cat((x, y)) for x, y in zip(data, public_data)]
            optimizer.zero_grad()
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs.view(-1), labels)
            loss.backward()
            optimizer.step()

    print(
        f"Epoch: {epoch} Private Train Loss: {test(model, private_train_loader, criterion, args):.10f}")


def experiment(args):
    setup_seed(args.seed)
    device = torch.device(args.device)

    # get multiple dataloader
    public_train_loader, private_train_loader, val_loader = select_train_valid_loader_full_batch(
        args)
    test_loader = select_test_loader_full_batch(args)

    if args.warm_start:

        st = time.process_time()
        model = select_model(args)
        criterion = nn.MSELoss()
        if args.pretrain_model is None:
            model.to(args.device)
            public_x, public_y = next(iter(public_train_loader))
            public_y = public_y.reshape((len(public_y), 1))
            optimal_theta = torch.linalg.solve(
                public_x.T @ public_x, public_x.T) @ public_y
            optimal_theta = optimal_theta.reshape(1, len(optimal_theta))
            with torch.no_grad():
                model.linear.weight = nn.Parameter(optimal_theta)
        else:
            model.load_state_dict(torch.load(args.pretrain_model))
        et = time.process_time()

        with torch.no_grad():
            data = next(iter(private_train_loader))
            public_data = next(iter(public_train_loader))
            data = [torch.cat((x, y)) for x, y in zip(data, public_data)]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            train_loss = criterion(outputs.view(-1), labels).item()
        print(
            f"Warm Start Loss using public data:")
        # print(
        #     f"private train loss:{test(model, private_train_loader, criterion, args) :.10f}"
        #     f"/validate loss: {test(model, val_loader, criterion, args) :.10f}"
        #     f"/combined train loss: {train_loss:.10f}/test loss: {test(model, test_loader, criterion, args) :.10f} "
        # )
        test_loss = test(model, test_loader, criterion, args)
        valid_loss = test(model, val_loader, criterion, args)
        private_train_loss = test(model, private_train_loader, criterion, args)

        data = {
            'throw_away_sgd': {
                'valid_loss': valid_loss,
                'private_train_loss': private_train_loss,
                'combined_loss': train_loss,
                'test_loss': test_loss,
                'cpu_time': et - st,
                'best_params': {
                    'lr': 0,
                    'private_epochs': 0,
                    'semi_dp_beta': args.semi_dp_beta,
                    'public_private_ratio': args.public_private_ratio,
                    'linear_reg_p': args.linear_reg_p,
                    'epsilon': args.epsilon,
                }
            },
        }
        data_json_string = json.dumps(data, indent=4)
        print(data_json_string)
        if args.optimizer == 'throw_away_sgd':
            return valid_loss, private_train_loss, train_loss, test_loss
    else:
        # cold start, get model
        model = select_model(args)
        with torch.no_grad():
            model.linear.weight = nn.Parameter(torch.zeros(
                model.linear.weight.shape, device=device))
        model.to(device)

    # init sgd optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer_down_cast = select_optimizer(args)

    # loss function
    criterion = nn.MSELoss()

    # init privacy engine
    privacy_engine = PrivacyEngine()
    model, optimizer, private_train_loader = privacy_engine.make_private_with_epsilon(module=model,
                                                                                      optimizer=optimizer,
                                                                                      data_loader=private_train_loader,
                                                                                      target_epsilon=args.epsilon,
                                                                                      target_delta=args.delta,
                                                                                      epochs=args.private_epochs,
                                                                                      max_grad_norm=args.max_grad_norm,
                                                                                      )
    if args.optimizer == 'semi_dp':
        print(
            f"std of Noise: {optimizer.max_grad_norm * optimizer.noise_multiplier: .3f} and C={optimizer.max_grad_norm}")
    else:
        print(
            f"std of Noise: {optimizer.max_grad_norm * optimizer.noise_multiplier: .3f} and C={optimizer.max_grad_norm}"
        )
    st = time.process_time()
    if args.optimizer == 'pda_pdmd_linear':
        X_pub, y_pub = next(iter(public_train_loader))
        public_sample_size = X_pub.shape[0]
        pub_mat = torch.mm(X_pub.T, X_pub) / public_sample_size + args.pda_pdmd_constant / args.linear_reg_p * torch.eye(
            args.linear_reg_p)
        w, v = torch.linalg.eig(pub_mat)
        # print(f"max eigen value: {max(w.real)}")
        # print(f"min eigen value: {min(w.real)}")
        min_eig = min(w.real)
        pub_mat = pub_mat / min_eig
        pub_mat_inv = torch.inverse(pub_mat)
        # pub_mat_inv = torch.eye(args.linear_reg_p) / public_sample_size
        optimizer = optimizer_down_cast.cast(
            optimizer, pub_mat_inv=pub_mat_inv)
    elif args.optimizer == 'semi_dp':
        optimizer = optimizer_down_cast.cast(
            optimizer, args.semi_dp_beta, clip_norm_public=args.semi_dp_public_norm)
    else:
        optimizer = optimizer_down_cast.cast(optimizer)
    et = time.process_time()
    elapsed_time_1 = et - st

    st = time.process_time()
    for epoch in range(args.private_epochs):
        train_per_epoch(epoch,
                        model,
                        public_train_loader,
                        private_train_loader,
                        criterion,
                        optimizer,
                        args,
                        )
        with torch.no_grad():
            data = next(iter(private_train_loader))
            public_data = next(iter(public_train_loader))
            data = [torch.cat((x, y)) for x, y in zip(data, public_data)]
            optimizer.zero_grad()
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            criterion(outputs.view(-1), labels).item()

        # # Early stopping
        # if test(model, val_loader, criterion, args) > 0.5:
        #     wandb.log({"valid_loss":  0.5,
        #                "private_train_loss": 0.5,
        #                "combined_loss": 0.5,
        #                "test_loss": 0.5
        #                })
        #     return 0.5, 0.5, 0.5, 0.5
    et = time.process_time()
    elapsed_time_2 = et - st
    print(f"Total time for training: {elapsed_time_1 + elapsed_time_2 : .10f}")

    with torch.no_grad():
        data = next(iter(private_train_loader))
        public_data = next(iter(public_train_loader))
        data = [torch.cat((x, y)) for x, y in zip(data, public_data)]
        optimizer.zero_grad()
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        train_loss = criterion(outputs.view(-1), labels).item()
    test_loss = test(model, test_loader, criterion, args)
    valid_loss = test(model, val_loader, criterion, args)
    private_train_loss = test(model, private_train_loader, criterion, args)
    wandb.log({"valid_loss":  valid_loss,
               "private_train_loss": private_train_loss,
               "combined_loss": train_loss,
               "test_loss": test_loss
               })
    print(
        f"Epoch:{epoch}/private train loss:{private_train_loss :.10f}"
        f"/validate loss: {test_loss:.10f}"
        f"/combined train loss: {train_loss:.10f}/test loss: {test_loss :.10f} "
    )
    data = {
        args.optimizer: {
            'valid_loss': valid_loss,
            'private_train_loss': private_train_loss,
            'combined_loss': train_loss,
            'test_loss': test_loss,
            'cpu_time': elapsed_time_1 + elapsed_time_2,
            'best_params': {
                'lr': args.lr,
                'private_epochs': args.private_epochs,
                'semi_dp_beta': args.semi_dp_beta,
                'public_private_ratio': args.public_private_ratio,
                'linear_reg_p': args.linear_reg_p,
                'epsilon': args.epsilon,
            }
        },
    }
    data_json_string = json.dumps(data, indent=4)
    print(data_json_string)
    return valid_loss, private_train_loss, train_loss, test_loss


def main():
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
