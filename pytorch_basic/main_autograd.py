from __future__ import print_function # Should comes first than torch
import torch
from torch.autograd import Variable

##
## Autograd.Variable is the central class of the package. It wraps a Tensor, and supports nearly all of operations defined on it. Once you finish your computation you can call .backward() and have all the gradients computed automatically!!!
## from: http://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

## ------------- Simple Variable ------------- ##
print("# ------------- Simple Variable ------------- #")
x_tensor = torch.ones(2, 2)
x = Variable(x_tensor, requires_grad=True)
print(x)

y = x + 2
print(y)

# y was created as a result of an operation, so it has a grad_fn.
print (y.grad_fn)

z = y * y * 3 # z = 3*y^2
out = z.mean()

print("z = y * y * 3\n", z, out)

## ------------- Simple Gradients ------------- ##
print("# ------------- Simple Gradients ------------- #")
# out.backward() is equivalent to doing out.backward(torch.Tensor([1.0]))
out.backward()
print("dout/dx \n", x.grad) # gradient of z = 3(x+2)^2, dout/dx = 3/2(x+2), x=1

## ------------- Crazy Gradients ------------- ##
print("# ------------- Crazy Gradients ------------- #")
x = torch.randn(3)
x = Variable(x, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print("y \n", y)

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)
print("dy/dx \n", x.grad)
