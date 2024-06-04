import torch
from torchvision import transforms, datasets
from torchvision.transforms import Normalize, ToTensor
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision.transforms.functional import to_tensor
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


class EMNIST_balanced(datasets.EMNIST):

    def _load_data(self):
        images = read_image_file(self.images_file)
        labels = read_label_file(self.labels_file)

        N = len(images)
        idx = torch.randperm(N)

        images = images[idx]
        labels = labels[idx]

        return images, labels

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img[None, :, :], int(target)


def get_emnist_dataset(args, is_train=True):
    dataset = EMNIST_balanced(root=args.data_dir,
                              split="bymerge",
                              download=True,
                              train=is_train,
                              )
    device = torch.device(args.device)
    dataset.data = torch.permute(to_tensor(dataset.data.numpy()).to(device), (1, 2, 0))
    return dataset
