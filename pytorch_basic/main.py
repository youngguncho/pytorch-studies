from __future__ import print_function # Should comes first than torch
import torch
import torchvision
import torch.nn as nn
import numpy as np

# ------------- Simple Matrix Generation ------------- #
print("# ------------- Simple Matrix Generation ------------- #")
# Generate 5x3 matrix w/o initialization
x = torch.Tensor(5, 2)
print(x)

# Generate 5x3 matrix with random elements
x = torch.rand(5, 2)
print(x)

# ------------- Simple Matrix Operation ------------- #
print("# ------------- Simple Matrix Operation ------------- #")
y = torch.rand(5, 2)
print(x+y)
print(torch.add(x, y))

# use result as argument
result = torch.Tensor(5, 2) # preallocation
torch.add(x, y, out=result)
print(result)

result2 = x + y # Automatic allocation
print(result2)

# inplace addition
y.add_(x)
print(y)

# ------------- Simple Matrix Row/Cols Operation ------------- #
print("# ------------- Simple Matrix Row/Cols Operation ------------- #")
x_first_col = x[:, 0]
print(x)
print(x_first_col)

print(x.size())
z1 = x.view(x.size(0)*x.size(1))
z2 = x.view(-1, 5) # '-1' represents Automatic size allocation with '2'
print(z1)
print(z2)

# ------------- Pytorch Tensor to Numpy ------------- #
print("# ------------- Pytorch Tensor to Numpy ------------- #")
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# Check numpy memory link
a.add_(1)
print(a)
print(b)
