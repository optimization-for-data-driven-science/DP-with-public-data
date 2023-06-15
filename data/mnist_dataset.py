import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
# from torchvision.datasets.mnist import read_image_file, read_label_file
# from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Normalize, ToTensor
from torchvision import transforms
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class MNIST_balanced(Dataset):

    def __init__(self, root="./data", is_train=True, device=torch.device("cpu"), transform=None):

        dset_tmp = datasets.FashionMNIST(root=root,
                                    train=is_train,
                                    download=True,
                                    transform=transform,
                                    )

        loader_tmp = DataLoader(dset_tmp, batch_size=len(
            dset_tmp), shuffle=True, num_workers=4)

        self.image, self.label = next(iter(loader_tmp))
        self.image, self.label = self.image.to(device), self.label.to(device)

    def __getitem__(self, index):

        return self.image[index], self.label[index]

    def __len__(self):
        return len(self.image)


def get_mnist_dataset(args, is_train=True):

    transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    dset = MNIST_balanced(root=args.data_dir,
                            is_train=is_train,
                            device=torch.device(args.device),
                            transform=transform
                            )
    return dset