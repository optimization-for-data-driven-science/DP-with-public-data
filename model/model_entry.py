from .emnist_conv_v1 import EmnistConvV1
from .emnist_conv_v2 import EmnistConvV2
from .cifar_10 import CifarConv, CifarConv2, ResNet18, CifarCvx, WRN_16_4
from .linear_reg import LinearRegModule
from .mnist import MNISTCvx
import torch.nn as nn

# import torchvision.models as models

model_choice = [
    "emnist_conv_v1",
    "emnist_conv_v2",
    "mnist_cvx",
    "linear_reg",
    "cifar_10_conv",
    "cifar_10_conv2",
    "cifar_10_cvx",
    "cifar_10_resnet",
    "cifar_10_wide_resnet",
]


def select_model(args):
    type2model = {
        "emnist_conv_v1": EmnistConvV1,
        "emnist_conv_v2": EmnistConvV2,
        "mnist_cvx": MNISTCvx,
        "linear_reg": LinearRegModule,
        "cifar_10_conv": CifarConv,
        "cifar_10_conv2": CifarConv2,
        "cifar_10_cvx": CifarCvx,
        "cifar_10_resnet": ResNet18,
        "cifar_10_wide_resnet": WRN_16_4,
    }

    model = type2model[args.model_type]
    if args.model_type == "linear_reg":
        model = model(args)
    else:
        model = model()
    return model


def equip_multi_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus)
    return model
