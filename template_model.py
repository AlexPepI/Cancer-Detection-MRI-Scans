import os
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['MLP', 'Inception3', 'inception_v3', 'End2EndModel']

model_urls = {
    # Downloaded inception model (optional)
    'downloaded': 'pretrained/inception_v3_google-1a9a5a14.pth',
    # Inception v3 ported from TensorFlow
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}




# class MLP(nn.Module):
#     def __init__(self, input_dim, num_classes, expand_dim):
#         super(MLP, self).__init__()
#         self.expand_dim = expand_dim
#         if self.expand_dim:
#             self.linear1 = nn.Linear(input_dim, expand_dim)
#             self.activation = torch.nn.ReLU()
#             self.linear2 = nn.Linear(expand_dim, num_classes) #softmax is automatically handled by loss function
#         else:
#             self.linear = nn.Linear(input_dim, num_classes)

#     def forward(self, x):
#         if self.expand_dim:
#             x = self.activation(self.linear1(x))
#             x = self.linear2(x)
#         else:
#             x = self.linear(x)
#         return x
    

# class MLP(nn.Module):
#     def __init__(self, input_dim, num_classes, expand_dim=None):
#         super().__init__()
#         if expand_dim:  # use a hidden layer
#             self.net = nn.Sequential(
#                 nn.Linear(input_dim, expand_dim),
#                 nn.ReLU(),
#                 nn.Linear(expand_dim, num_classes)
#             )
#         else:  # just a single linear classifier
#             self.net = nn.Linear(input_dim, num_classes)

#     def forward(self, x):
#         return self.net(x)


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, expand_dim=None):
        super().__init__()
        if expand_dim:  # use a hidden layer
            self.net = nn.Sequential(
                nn.Linear(input_dim, expand_dim),
                nn.ReLU(),
                nn.Linear(expand_dim, expand_dim), ## added another hidden layer. 1 -> 16 -> 16 -> 1
                nn.ReLU(),
                # nn.Linear(expand_dim + 64, expand_dim), ## added another hidden layer. 1 -> 16 -> 16 -> 1
                # nn.ReLU(),
                nn.Linear(expand_dim, num_classes)
            )
        else:  # just a single linear classifier
            self.net = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.net(x)