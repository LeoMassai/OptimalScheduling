import numpy as np
import itertools
from scipy.stats import expon
from Hypoexp import Hypoexponential
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(DeepNeuralNetwork, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        # Define the forward pass
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for binary output
        return x

    def predict_binary(self, x, threshold=0.5):
        # Apply threshold for binary prediction during inference
        with torch.no_grad():
            output = self.forward(x)
            binary_output = (output > threshold).float()
        return binary_output


def get_permutations_and_matrices(input_vector):
    permutations_list = list(itertools.permutations(input_vector))
    matrices_list = [np.zeros((len(input_vector), len(input_vector)), dtype=int) for _ in permutations_list]

    for i, perm in enumerate(permutations_list):
        for j, idx in enumerate(perm):
            matrices_list[i][j, idx - 1] = 1  # Fix: Subtract 1 from the index

    return permutations_list, matrices_list


def probwinning(input_vector):
    n = len(input_vector)
    output_vector = [0] * n

    for k in range(n):
        product = 1
        for i in range(k - 1, -1, -1):
            product *= (1 - input_vector[i])
        output_vector[k] = input_vector[k] * product

    return output_vector


def times(iv, T):
    n = len(iv)
    ov = [0] * n
    for k in range(2, n + 1):
        Hy = Hypoexponential(1 / iv[range(k)])
        ov[k - 1] = (Hy.cdf([0, T]))[-1]
    ov[0] = expon.cdf(x=T, scale=iv[0])
    return ov


class scheduling:
    def __init__(self, r, p, t, T, q):
        self.r = r
        self.p = p
        self.t = t
        self.T = T
        self.q = q

    def optord(self):
        op = 0
        otp = 0
        maxr = 0
        n = len(self.r)
        per, perm = get_permutations_and_matrices([x for x in range(1, n + 1)])
        er = np.zeros(len(per))
        TP = np.zeros(len(per))
        for k, j in enumerate(perm):
            rp = self.r @ j
            pp = self.p @ j
            tp = self.t @ j
            gp = times(tp, self.T)
            w = probwinning(pp)
            f = 1 - sum(w)
            er[k] = rp @ w
            TP[k] = f * gp[-1] + np.array(w) @ np.array(gp)
            # Optimal solution
            if TP[k] > self.q:
                if er[k] > maxr:
                    maxr = er[k]
                    op = j
                    otp = TP[k]
        return op, otp, er, TP
