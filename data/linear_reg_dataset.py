from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import os


def _generate_x(p):
    first_part_features = int(p / 5)
    last_part_features = p - first_part_features

    nonzero_first_cnt = 40
    nonzero_last_cnt = 80

    nonzero_first_idx = torch.randperm(first_part_features)[:nonzero_first_cnt]
    nonzero_last_idx = torch.randperm(last_part_features)[:nonzero_last_cnt] + first_part_features

    x_i = torch.zeros(p)
    x_i[nonzero_first_idx] = 0.05
    x_i[nonzero_last_idx] = 0.05
    return x_i


class LinearRegDataset(Dataset):
    def __init__(self, args, is_train=True):
        self.device = args.device
        self.data_dir = os.path.join(args.data_dir, 'LINEAR_REG')
        self.linear_reg_data_dir = os.path.join(self.data_dir, "P-{}".format(args.linear_reg_p))
        self.is_train = is_train

        if not os.path.exists(self.linear_reg_data_dir):
            os.makedirs(self.linear_reg_data_dir)

        if self.is_train:
            self.filename_prefix = ""
        else:
            self.filename_prefix = "_test"

        self.X_file = os.path.join(self.linear_reg_data_dir, 'X' + self.filename_prefix + '.pt')
        self.y_file = os.path.join(self.linear_reg_data_dir, 'y' + self.filename_prefix + '.pt')
        self.y_opt_file = os.path.join(self.linear_reg_data_dir, 'y_opt' + self.filename_prefix + '.pt')
        self.theta_file = os.path.join(self.linear_reg_data_dir, 'theta' + '.pt')

        if os.path.exists(self.X_file):
            self.X = torch.load(self.X_file, map_location=self.device)
            self.y = torch.load(self.y_file, map_location=self.device)
            self.theta = torch.load(self.theta_file, map_location=self.device)
            if self.is_train:
                self.size = self.X.shape[0]
            else:
                self.size = 10000
        else:
            self.size = (10000 + args.linear_reg_ratio * args.linear_reg_p)
            print(f"Generating linear regression dataset with MSE 0.01 and p {args.linear_reg_p}")
            print("Taking roughly 1 minute to finish...")
            self.size = int(self.size / (1 - args.train_valid_ratio))  # generate additional data for validation
            if is_train:
                self.theta = torch.normal(
                    mean=0,
                    std=1,
                    size=(1, args.linear_reg_p),
                    device=args.device,
                )
            else:
                if not os.path.exists(self.theta_file):
                    raise ValueError("Have to generate Train dataset before generating test dataset!!!")
                self.theta = torch.load(self.theta_file, map_location=self.device)
            self.X = None
            for _ in tqdm(range(self.size)):
                x_i = _generate_x(args.linear_reg_p)
                if self.X is None:
                    self.X = x_i
                else:
                    self.X = torch.vstack((self.X, x_i))
            self.X = self.X.to(args.device)

            self.y = torch.mm(self.theta, self.X.T).to(args.device)
            self.y = self.y.to(args.device)
            torch.save(self.y, self.y_opt_file)
            self.y = torch.normal(
                mean=self.y,
                std=0.1 * torch.ones(self.y.shape, device=torch.device(args.device))
            )
            self.y = self.y.reshape(-1)
            torch.save(self.X, self.X_file)
            torch.save(self.y, self.y_file)
            if is_train:
                torch.save(self.theta, self.theta_file)

        print("X Shape: {}".format(self.X.shape))
        print("Y Shape: {}".format(self.y.shape))

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]
