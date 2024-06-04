import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, ToTensor
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.transforms.functional import to_tensor
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class CIFAR_10_balanced(Dataset):

    def __init__(self, root="./data", is_train=True, transform=None, device=torch.device("cpu")):

        dset_tmp = datasets.CIFAR10(root=root,
                                    train=is_train,
                                    transform=transform,
                                    download=True,
                                    )

        loader_tmp = DataLoader(dset_tmp, batch_size=len(
            dset_tmp), shuffle=True, num_workers=4)

        self.image, self.label = next(iter(loader_tmp))
        self.image, self.label = self.image.to(device), self.label.to(device)

    def __getitem__(self, index):

        return self.image[index], self.label[index]

    def __len__(self):
        return len(self.image)


def get_cifar_10_dataset(args, is_train=True, transform=None):
    if transform is None:
        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
        ])
    dset = CIFAR_10_balanced(root=args.data_dir,
                             transform=transform,
                             is_train=is_train,
                             device=torch.device(args.device)
                             )
    return dset
