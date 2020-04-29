import numpy as np
import torch


class ANETDataset:

    def __init__(self, anet_cases):
        self.x_train_tensor, self.y_train_tensor = self._init(anet_cases)

    def _init(self, anet_cases):
        # anet_cases = list of nodes

        x_train = []  # x = state
        y_train = []  # y = distribution

        for case in anet_cases:
            case_state, case_target_distribution = case.get_as_numpy_arrays()
            x_train.append(case_state)
            y_train.append(case_target_distribution)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        return torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float()
