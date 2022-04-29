import torch
from torch import nn
from torch.autograd import Function


class GRL(Function):

    @staticmethod
    def forward(ctx, input, alpha):
        ctx.alpha = alpha
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.alpha * grad_output.neg()
        return output, None


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(310, 310), nn.BatchNorm1d(310), nn.ReLU(), 
            nn.Linear(310, 128), nn.BatchNorm1d(128), nn.ReLU(), 
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), 
        )

    def forward(self, x):
        return self.encoder(x)

class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.encoder = Encoder()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 3)
        )
        self.discriminator = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
            # nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x, alpha=1.):
        feature = self.encoder(x)
        feature_reverse = GRL.apply(feature, alpha)
        return self.classifier(feature), self.discriminator(feature_reverse)


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.encoder = Encoder()
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), 
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))

