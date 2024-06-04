from data.mnist_dataset import get_mnist_dataset
from data.emnist_dataset import get_emnist_dataset
from data.linear_reg_dataset import LinearRegDataset
from data.linear_reg_dataset_experiment import LinearRegDatasetExp
from data.linear_reg_dataset_experiment_gaussian import LinearRegDatasetExpGaussian
from data.cifar10_dataset import get_cifar_10_dataset
from torch.utils.data import DataLoader, random_split

data_choice = ['linear_reg', 'mnist', 'emnist', 'linear_reg_exp',
               'linear_reg_exp_gaussian', 'cifar10']


def get_dataset_by_type(args, is_train=True, transform=None):
    if args.data_type == 'linear_reg':
        dataset = LinearRegDataset(args, is_train=is_train)
    elif args.data_type == 'mnist':
        dataset = get_mnist_dataset(args, is_train=is_train)
    elif args.data_type == 'emnist':
        dataset = get_emnist_dataset(args, is_train=is_train)
    elif args.data_type == 'linear_reg_exp':
        dataset = LinearRegDatasetExp(args, is_train=is_train)
    elif args.data_type == 'linear_reg_exp_gaussian':
        dataset = LinearRegDatasetExpGaussian(args, is_train=is_train)
    elif args.data_type == "cifar10":

        from torchvision import transforms

        if is_train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        dataset = get_cifar_10_dataset(
            args, is_train=is_train, transform=transform)
    return dataset


def select_train_valid_loader(args):
    # usually we need loader in training, and dataset in eval/test
    train_dataset = get_dataset_by_type(args, is_train=True)
    # use args.train_valid_ratio as train set, val set
    split_train_size = int((1 - args.train_valid_ratio) * (len(train_dataset)))
    split_valid_size = len(train_dataset) - split_train_size

    train_dataset, val_dataset = random_split(
        train_dataset, [split_train_size, split_valid_size])

    # use args.public_private_ratio as public train set, private train set
    split_public_size = int(args.public_private_ratio * (len(train_dataset)))
    split_private_size = len(train_dataset) - split_public_size

    public_train_dataset, private_train_dataset = random_split(train_dataset,
                                                               [split_public_size, split_private_size])
    if args.optimizer == "deep_mind":
        private_train_dataset = train_dataset

    print(
        f"public train set size: {len(public_train_dataset)}, "
        f"private train set size: {len(private_train_dataset)}, "
        f"validation set size: {len(val_dataset)}")

    public_train_loader = DataLoader(public_train_dataset,
                                     batch_size=args.pub_batch_size,
                                     shuffle=True,
                                     num_workers=args.data_n_worker,
                                     pin_memory=False,
                                     drop_last=args.drop_last)
    private_train_loader = DataLoader(private_train_dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.data_n_worker,
                                      pin_memory=False,
                                      drop_last=args.drop_last)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.data_n_worker,
                            pin_memory=False,
                            drop_last=args.drop_last)
    if args.optimizer == "deep_mind":
        return None, private_train_loader, val_loader
    return public_train_loader, private_train_loader, val_loader


def select_train_valid_loader_full_batch(args):
    # usually we need loader in training, and dataset in eval/test
    train_dataset = get_dataset_by_type(args, is_train=True)
    # use args.train_valid_ratio as train set, val set
    split_train_size = int((1 - args.train_valid_ratio) * (len(train_dataset)))
    split_valid_size = len(train_dataset) - split_train_size

    train_dataset, val_dataset = random_split(
        train_dataset, [split_train_size, split_valid_size])

    # use args.public_private_ratio as public train set, private train set
    if args.model_type == 'linear_reg' and args.data_type == 'linear_reg':
        split_public_size = int(args.linear_reg_ratio * args.linear_reg_p)
        split_private_size = int(len(train_dataset) - split_public_size)
    elif args.model_type == 'linear_reg' and args.data_type == 'linear_reg_exp':
        split_public_size = int(len(train_dataset) * args.public_private_ratio)
        split_private_size = len(train_dataset) - split_public_size
    else:
        split_public_size = int(
            args.public_private_ratio * (len(train_dataset)))
        split_private_size = len(train_dataset) - split_public_size

    public_train_dataset, private_train_dataset = random_split(train_dataset,
                                                               [split_public_size, split_private_size])
    # Full Batch Gradient
    if split_public_size > 0:
        public_train_loader = DataLoader(public_train_dataset,
                                         batch_size=split_public_size,
                                         shuffle=True,
                                         num_workers=args.data_n_worker)
    else:
        public_train_loader = None
    if split_private_size > 0:
        private_train_loader = DataLoader(private_train_dataset,
                                          batch_size=split_private_size,
                                          shuffle=False,
                                          num_workers=args.data_n_worker)
    else:
        private_train_loader = None
    val_loader = DataLoader(val_dataset,
                            batch_size=split_valid_size,
                            shuffle=True,
                            num_workers=args.data_n_worker)

    print(
        f"public train set size: {len(public_train_dataset)}, "
        f"private train set size: {len(private_train_dataset)}, "
        f"validation set size: {len(val_dataset)}")

    return public_train_loader, private_train_loader, val_loader


def select_train_valid_loader_linear_reg_batch(args):
    # usually we need loader in training, and dataset in eval/test
    train_dataset = get_dataset_by_type(args, is_train=True)
    # use args.train_valid_ratio as train set, val set
    split_train_size = int((1 - args.train_valid_ratio) * (len(train_dataset)))
    split_valid_size = len(train_dataset) - split_train_size

    train_dataset, val_dataset = random_split(
        train_dataset, [split_train_size, split_valid_size])

    # use args.public_private_ratio as public train set, private train set
    if args.model_type == 'linear_reg' and args.data_type == 'linear_reg':
        split_public_size = int(args.linear_reg_ratio * args.linear_reg_p)
        split_private_size = int(len(train_dataset) - split_public_size)
    elif args.model_type == 'linear_reg' and args.data_type == 'linear_reg_exp':
        split_public_size = int(len(train_dataset) * args.public_private_ratio)
        split_private_size = len(train_dataset) - split_public_size
    else:
        split_public_size = int(
            args.public_private_ratio * (len(train_dataset)))
        split_private_size = len(train_dataset) - split_public_size

    public_train_dataset, private_train_dataset = random_split(train_dataset,
                                                               [split_public_size, split_private_size])
    # Full Batch Gradient
    if split_public_size > 0:
        public_train_loader = DataLoader(public_train_dataset,
                                         batch_size=args.pub_batch_size,
                                         shuffle=True,
                                         num_workers=args.data_n_worker)
    else:
        public_train_loader = None
    if split_private_size > 0:
        private_train_loader = DataLoader(private_train_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=args.data_n_worker)
    else:
        private_train_loader = None
    val_loader = DataLoader(val_dataset,
                            batch_size=split_valid_size,
                            shuffle=True,
                            num_workers=args.data_n_worker)

    print(
        f"public train set size: {len(public_train_dataset)}, "
        f"private train set size: {len(private_train_dataset)}, "
        f"validation set size: {len(val_dataset)}")

    return public_train_loader, private_train_loader, val_loader


def select_test_loader_linear_reg_batch(args):
    test_dataset = get_dataset_by_type(args, is_train=False)
    print(f"test set size: {len(test_dataset)}")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=len(test_dataset),
                                 shuffle=True,
                                 num_workers=args.data_n_worker)
    return test_dataloader


def select_test_loader_full_batch(args):
    test_dataset = get_dataset_by_type(args, is_train=False)
    print(f"test set size: {len(test_dataset)}")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=len(test_dataset),
                                 shuffle=True,
                                 num_workers=args.data_n_worker)
    return test_dataloader


def select_test_loader(args):
    test_dataset = get_dataset_by_type(args, is_train=False)
    print(f"Test set size: {len(test_dataset)}")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 num_workers=args.data_n_worker)
    return test_dataloader
