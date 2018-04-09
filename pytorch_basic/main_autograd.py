from __future__ import print_function # Should comes first than torch
import torch
from torch.autograd import Variable

##
## Autograd.Variable is the central class of the package. It wraps a Tensor, and supports nearly all of operations defined on it. Once you finish your computation you can call .backward() and have all the gradients computed automatically!!!
## from: http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

# ------------- Simple Gradient ------------- #
print("# ------------- Simple Gradient ------------- #")
x_tensor = torch.ones(2, 2)
x = Variable(x, requires_grad=True)

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

# ------------- Pytorch Cuda Tensor ------------- #
print("# ------------- Pytorch Cuda Tensor ------------- #")
if torch.cuda.is_available():
    xc = x.cuda()
    yc = y.cuda()
    zc = xc + yc
    print(zc)
    print(x+y)
