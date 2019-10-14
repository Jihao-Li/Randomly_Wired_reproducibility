import torch


a = torch.ones(2)
b = torch.rand(1, 2, 3, 3)
c = torch.rand(1, 2, 4, 4)
d = [b, c]
print(d)

for e, f in zip(a, d):
    print(e)
    print(f)
    h = e * f
    print(h)
