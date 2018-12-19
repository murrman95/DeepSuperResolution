import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        #
        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9),
            nn.ReLU(),
            nn.Conv2d(64,32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(1)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        H = self.feature_extractor_part1(x)
        #H = H.view(-1, 50 * 4 * 4)
        #H = self.feature_extractor_part2(H)  # NxL

        #A = self.attention(H)  # NxK
        #A = torch.transpose(A, 1, 0)  # KxN
        #A = F.softmax(A, dim=1)  # softmax over N

        ##Matrix multiplication
        #M = torch.mm(A, H)  # KxL

        #Y_prob = self.classifier(M)
        #Y_hat = torch.ge(Y_prob, 0.5).float()

        return H

    # AUXILIARY METHODS
    def calculate_SR_error(self, X, Y):
        Y = Y.float()
        HR_Gen = self.forward(X)
        error = sum(abs(X - HR_Gen))

        return error

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
