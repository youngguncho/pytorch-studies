from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='Pytorch Vanilla GAN Exmaple')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                    help='batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs on training (default: 10)')
parser.add_argument('--g-lr', type=float, default=0.0003, metavar='LR',
                    help='generator learning rate (default: 0.001)')
parser.add_argument('--d-lr', type=float, default=0.0003, metavar='LR',
                    help='discriminator learning rate (default: 0.001)')
parser.add_argument('--latent-size', type=int, default=64, metavar='L',
                    help='the length of latent vector z (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

# args for variables
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                    std=(0.5, 0.5, 0.5))])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                    transform = transform),
    batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True,
                    transform = transform),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(latent_size + 10, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 784),
                        nn.Tanh()
                        )

    def forward(self, x):
        batch_size = x.size(0)
        output =  self.model(x)
        output = output.view(batch_size, 1, 28, 28)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
                        nn.Linear(28*28 + 10, 256), # mnist image: 28x28
                        nn.LeakyReLU(0.2),
                        nn.Linear(256, 256),
                        nn.LeakyReLU(0.2),
                        nn.Linear(256, 1),
                        nn.Sigmoid()
                        )

    def forward(self, x):
        # batch_size = x.size(0)
        # x = x.view(batch_size, -1)
        return self.model(x)


G = Generator(args.latent_size)
D = Discriminator()

if args.cuda:
    G.cuda()
    D.cuda()

def to_var(x):
    if args.cuda:
        x = x.cuda()
    return Variable(x)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def make_one_hot(batch_size, labels):
    one_hot = torch.FloatTensor(batch_size, 10)
    one_hot.zero_()
    one_hot.scatter_(1, labels.unsqueeze(1), 1)
    return to_var(one_hot)

# optimizer_G = optim.SGD(G.parameters(), lr = args.g_lr)
# optimizer_D = optim.SGD(D.parameters(), lr = args.d_lr)

optimizer_G = optim.Adam(G.parameters(), lr = args.g_lr)
optimizer_D = optim.Adam(D.parameters(), lr = args.d_lr)

def train(epoch):
    G.train()
    D.train()

    for batch_idx, (data, label) in enumerate(train_loader):


        batch_size = data.size(0)
        data = to_var(data)

        # Create labels to compute adversarial loss
        real_labels = to_var(torch.ones(batch_size))
        fake_labels = to_var(torch.zeros(batch_size))

        ## -------------- Train discriminator -------------- ##

        one_hot = make_one_hot(batch_size, label)

        x = data.view(batch_size, -1)
        x_input = torch.cat((x, one_hot), dim=1)
        real_outputs = D(x_input)

        z_label = (torch.rand(batch_size)*10).type(torch.LongTensor)
        one_hot = make_one_hot(batch_size, z_label)
        z = to_var(torch.randn(batch_size, args.latent_size))
        z_input = torch.cat((z, one_hot), dim=1)

        fake_data = G(z_input)

        x_fake = torch.cat((fake_data.view(batch_size, -1), one_hot), dim=1)
        fake_outputs = D(x_fake)

        d_loss_real = F.binary_cross_entropy(real_outputs.squeeze(1), real_labels)
        d_loss_fake = F.binary_cross_entropy(fake_outputs.squeeze(1), fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss_print = d_loss.data[0]

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        ## -------------- Train Generator -------------- ##
        z = to_var(torch.randn(batch_size, args.latent_size))
        one_hot = one_hot.index_select(0, to_var(torch.randperm(batch_size))) # Randomly permute one_hot (same as random label)
        z_input = torch.cat((z, one_hot), dim=1)

        fake_data = G(z_input)
        x_fake = torch.cat((fake_data.view(batch_size, -1), one_hot), dim=1)
        fake_outputs = D(x_fake)
        g_loss_fake = F.binary_cross_entropy(fake_outputs.squeeze(1), real_labels)

        g_loss = g_loss_fake
        g_loss_print = g_loss.data[0]

        images = fake_data

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]: D_loss: {} / G_loss: {}'.format(epoch, batch_idx * batch_size, len(train_loader.dataset), d_loss_print, g_loss_print))


    z_test = to_var(torch.randn(args.test_batch_size, args.latent_size))
    num_test_each = args.test_batch_size // 10
    test_labels = torch.from_numpy(np.arange(10))
    test_labels = test_labels.repeat(10)
    test_one_hot = make_one_hot(args.test_batch_size, test_labels)
    z_test = torch.cat((z_test, test_one_hot), dim=1)
    test_images = G(z_test)



    save_image(denorm(test_images.data), './samples/test_{}_epoch.png'.format(epoch), nrow=num_test_each)
    save_image(denorm(images.data), './samples/fake_{}_epoch.png'.format(epoch), nrow=8)
    save_image(denorm(data.data), './samples/real_image.png', nrow=8)






for epoch in range(1, args.epochs+1):
    train(epoch)

torch.save(D.state_dict(), './models/d_model.pth')
torch.save(G.state_dict(), './models/g_model.pth')
