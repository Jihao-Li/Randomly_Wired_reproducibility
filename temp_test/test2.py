import torch


conv1 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
a = torch.rand(1, 3, 56, 56)
b = conv1(a)
print(b)
