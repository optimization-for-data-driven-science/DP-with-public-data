import torch
from torch import optim
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import torch.nn as nn
from model.model_entry import select_model
from data.data_entry import select_train_valid_loader, select_test_loader
from optimizer.optimizer_entry import select_optimizer
from utils import setup_seed
from options import parse_train_args
import argparse
import wandb
from torch.optim.lr_scheduler import ExponentialLR, StepLR, CosineAnnealingLR, ReduceLROnPlateau, CyclicLR
import warnings
from train.privacy_engine_extend import PrivacyEngineExtend
import time
import json


def inf_loader(loader):
    while True:
        for item in loader:
            yield item


def train_per_iter(
        current_iter,
        model,
        public_train_loader,
        private_train_loader,
        val_loader,
        criterion,
        privacy_engine,
        optimizer,
        args,
):
    device = torch.device(args.device)

    model.train()
    correct = 0
    total = 0
    train_loss = 0
    if args.drop_last is None:
        warnings.warn(
            "Data loader drop last disabled. Consider enable for semi-dp traning.")

    #######################################################
    #   Training Starts
    #######################################################

    # ------- public update --------
    if args.optimizer in ['semi_dp', 'pda_pdmd', 'throw_away_sgd']:
        optimizer.zero_grad()
        inputs, labels = next(public_train_loader)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step_public()

    # ------- private update --------
    if args.optimizer in ['semi_dp', 'pda_pdmd', 'deep_mind', 'dp_sgd']:
        optimizer.zero_grad()
        inputs, labels = next(private_train_loader)
        inputs, labels = inputs.to(device), labels.to(device)
        # if args.optimizer == 'deep_mind' or args.optimizer == 'dp_sgd':
        #     public_inputs, public_labels = next(iter(public_train_loader))
        #     public_inputs = public_inputs.to(device)
        #     public_labels = public_labels.to(device)
        #     inputs = torch.cat((inputs, public_inputs))
        #     labels = torch.cat((labels, public_labels))
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # ------- metric --------
    with torch.no_grad():
        # grab prediction as one-dimensional tensor
        prediction = outputs.argmax(dim=-1)

    correct = (prediction == labels).sum().item()
    train_loss = loss.item()
    train_acc = 100 * correct / labels.size(0)

    if (current_iter + 1) % 100 == 0:
        epochs = (current_iter + 1) // 100
        print(
            f"Epoch: {epochs}\n"
            f"Training Loss: {train_loss :.10f} "
            f"Training accuracy: {train_acc :.3f}% "
        )
        wandb.log({"Epoch": epochs,
                   "Training Loss": train_loss,
                   "Training accuracy": train_acc
                   })

    #######################################################
    #   Training Ends
    #######################################################

    #######################################################
    #   Validation Starts
    #######################################################

    if (current_iter + 1) % 100 == 0:
        model.eval()
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                prediction = outputs.argmax(dim=-1)
                total += labels.size(0)
                correct += (prediction == labels).sum().item()

            val_loss = val_loss / total
            val_acc = 100 * correct / total
        print(
            f"\t Validation loss: {val_loss :.10f} "
            f"Validation accuracy: {val_acc :.3f}%"
        )
        wandb.log({"Validation loss": val_loss,
                   "Validation accuracy": val_acc,
                   })

        #######################################################
        #   Validation Ends
        #######################################################


def test(model, test_loader, criterion, args):
    model.eval()
    device = torch.device(args.device)

    correct = 0
    total = 0
    test_loss = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * labels.size(0)
            prediction = outputs.argmax(dim=-1)
            total += labels.size(0)
            correct += (prediction == labels).sum().item()

        test_loss = test_loss / total
        test_acc = 100 * correct / total
        print(
            f"******** Test loss: {test_loss :.10f} ********\n"
            f"******** Test accuracy: {test_acc :.4f} ********"
        )
        wandb.log({"Test accuracy": test_acc,
                   "Test loss": test_loss,
                   })
    return test_loss, test_acc


def experiment(args):
    setup_seed(args.seed)
    device = torch.device(args.device)

    # get model
    model = select_model(args)
    model = ModuleValidator.fix(model)

    # load pretrain warm-up model
    st = time.process_time()
    if args.pretrain_model is not None:
        model.load_state_dict(torch.load(args.pretrain_model))
    elif args.warm_start:
        raise NotImplementedError
    elapsed_time_1 = time.process_time() - st

    # get multiple dataloader
    public_train_loader, private_train_loader, val_loader = select_train_valid_loader(
        args)
    test_loader = select_test_loader(args)

    model.to(device)

    # init sgd optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    optimizer_down_cast = select_optimizer(args)

    # loss function
    criterion = nn.CrossEntropyLoss()

    private_N_iters = len(private_train_loader)

    # init privacy engine
    privacy_engine = PrivacyEngine()
    private_engine = PrivacyEngineExtend.cast(privacy_engine)

    # Using accountant of Opacus but prscribed noise based on steps instead of epochs
    # Assume public batch size equals to private batch size
    model, optimizer, private_train_loader = private_engine.make_private_with_steps(module=model,
                                                                                    optimizer=optimizer,
                                                                                    data_loader=private_train_loader,
                                                                                    target_epsilon=args.epsilon,
                                                                                    target_delta=args.delta,
                                                                                    poisson_sampling=False,
                                                                                    sample_rate=1 / private_N_iters,
                                                                                    steps=args.iterations,
                                                                                    max_grad_norm=args.max_grad_norm,
                                                                                    )

    private_train_loader = inf_loader(private_train_loader)
    public_train_loader = inf_loader(public_train_loader)

    if args.optimizer == 'semi_dp':
        print(
            f"std of Noise: {optimizer.max_grad_norm * optimizer.noise_multiplier: .3f} and C={optimizer.max_grad_norm}"
        )
    else:
        print(
            f"std of Noise: {optimizer.max_grad_norm * optimizer.noise_multiplier: .3f} and C={optimizer.max_grad_norm}"
        )

    if args.optimizer == 'pda_pdmd':
        raise NotImplementedError
    elif args.optimizer == 'semi_dp':
        optimizer = optimizer_down_cast.cast(
            optimizer, args.semi_dp_beta, clip_norm_public=args.semi_dp_public_norm)
    else:
        optimizer = optimizer_down_cast.cast(optimizer)

    scheduler = None

    if args.lr_scheduler == 'explr':
        scheduler = ExponentialLR(optimizer, gamma=args.lr_scheduler_gamma)
    elif args.lr_scheduler == 'steplr':
        scheduler = StepLR(
            optimizer, step_size=args.lr_scheduler_step_size, gamma=args.lr_scheduler_gamma)
    elif args.lr_scheduler == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.lr_scheduler_T_max, eta_min=args.lr_scheduler_eta_min)
    elif args.lr_scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_scheduler_factor,
                                      patience=args.lr_scheduler_patience, verbose=True)
    elif args.lr_scheduler == 'cycle':
        scheduler = CyclicLR(optimizer, base_lr=args.lr_scheduler_base_lr, max_lr=args.lr_scheduler_max_lr,
                             step_size_up=args.lr_scheduler_step_size_up, mode=args.lr_scheduler_mode)

    print(f"Total Number of Iterations: {args.iterations}")
    print(f"Total Number of Epochs: {args.iterations // 100}")

    st = time.process_time()
    for iters in range(args.iterations):
        if (iters + 1) % 100 == 0:
            if scheduler is not None:
                if args.lr_scheduler != "plateau":
                    print(f'Current Learning Rate: {scheduler.get_last_lr()}')
        train_per_iter(iters,
                       model,
                       public_train_loader,
                       private_train_loader,
                       val_loader,
                       criterion,
                       privacy_engine,
                       optimizer,
                       args,
                       )

        if (iters + 1) % 100 == 0:
            test_loss, test_acc = test(model, test_loader, criterion, args)

            if scheduler is not None:
                if args.lr_scheduler != "plateau":
                    scheduler.step()
                else:
                    scheduler.step(test_loss)
    et = time.process_time()
    elapsed_time_2 = et - st

    # test model
    valid_loss, valid_acc = test(model, val_loader, criterion, args)
    test_loss, test_acc = test(model, test_loader, criterion, args)

    data = {
        args.optimizer: {
            'valid_loss': valid_loss,
            'valid_acc': valid_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'cpu_time': elapsed_time_1 + elapsed_time_2,
            'best_params': {
                'lr': args.lr,
                'iterations': args.iterations,
                'batch_size': args.batch_size,
                'pub_batch_size': args.pub_batch_size,
                'semi_dp_beta': args.semi_dp_beta,
                'public_private_ratio': args.public_private_ratio,
                'epsilon': args.epsilon,
            }
        },
    }
    data_json_string = json.dumps(data, indent=4)
    print(data_json_string)

    return valid_loss, valid_acc, test_loss, test_acc


def main():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    args = parser.parse_args()
    if not args.use_wandb:
        wandb.init(project=args.project_name, config=args, mode="disabled")
    else:
        wandb.init(project=args.project_name, config=args)
    print(f"Running on {args.device}")
    experiment(args)


if __name__ == '__main__':
    main()
