import torch


class ANETDataset:

    def __init__(self, anet_cases):
        self.x_train_tensor, self.y_train_tensor = self._init(anet_cases)

    def _init(self, anet_cases):
        x_train = []
        y_train = []

        for case in anet_cases:
            x_train.append(case.x_input)
            y_train.append(case.target_distribution)

        return torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()