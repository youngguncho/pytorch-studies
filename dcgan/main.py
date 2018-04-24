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
parser = argparse.ArgumentParser(description='Pytorch Deep Convolutional GAN Exmaple')
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
parser.add_argument('--latent-size', type=int, default=100, metavar='L',
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

# custom weight initialization
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, latent_size):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
                        # input z
                        # state_size = latent_size x 1 x 1
                        nn.ConvTranspose2d(latent_size, 512, 4, 1, 0, bias=False),
                        nn.BatchNorm2d(512),
                        nn.LeakyReLU(0.2, inplace=True),
                        # state_size = 512 x 4 x 4
                        nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(256),
                        nn.LeakyReLU(0.2, inplace=True),
                        # state_size = 256 x 8 x 8
                        nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                        nn.BatchNorm2d(128),
                        nn.LeakyReLU(0.2, inplace=True),
                        # state_size = 128 x 16 x 16
                        nn.ConvTranspose2d(128, 1, 4, 2, 3, bias=False),
                        nn.Tanh()
                        # state_size = 1 x 28 x 28
                        )

    def forward(self, x):
        output =  self.model(x)
        return output

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
                        # input x
                        # state_sizs = 1 x 28 x 28
                        nn.Conv2d(1, 128, 4, 2, 1, bias=False),
                        nn.LeakyReLU(0.2, inplace=True),
                        # state_sizs = 128 x 14 x 14
                        nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                        nn.LeakyReLU(0.2, inplace=True),
                        # state_sizs = 256 x 7 x 7
                        nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                        nn.LeakyReLU(0.2, inplace=True),
                        # state_sizs = 512 x 3 x 3
                        nn.Conv2d(512, 1, 3, 1, 0, bias=False),
                        nn.Sigmoid()
                        # state_sizs = 1 x 1 x 1
                        )

    def forward(self, x):
        # batch_size = x.size(0)
        # x = x.view(batch_size, -1)
        output = self.model(x)
        return output.view(-1, 1).squeeze(1)


G = Generator(args.latent_size)
D = Discriminator()
G.apply(weight_init)
D.apply(weight_init)


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
        z = to_var(torch.randn(batch_size, args.latent_size, 1, 1))

        real_outputs = D(data)

        fake_data = G(z)
        fake_outputs = D(fake_data)

        d_loss_real = F.binary_cross_entropy(real_outputs, real_labels)
        d_loss_fake = F.binary_cross_entropy(fake_outputs, fake_labels)

        d_loss = d_loss_real + d_loss_fake
        d_loss_print = d_loss.data[0]

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        ## -------------- Train Generator -------------- ##
        z = to_var(torch.randn(batch_size, args.latent_size, 1, 1))

        fake_data = G(z)
        fake_outputs = D(fake_data)
        g_loss_fake = F.binary_cross_entropy(fake_outputs, real_labels)

        g_loss = g_loss_fake
        g_loss_print = g_loss.data[0]

        images = fake_data

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]: D_loss: {} / G_loss: {}'.format(epoch, batch_idx * batch_size, len(train_loader.dataset), d_loss_print, g_loss_print))


    # z_test = to_var(torch.randn(args.test_batch_size, args.latent_size))
    # num_test_each = args.test_batch_size // 10
    # test_labels = torch.from_numpy(np.arange(10))
    # test_labels = test_labels.repeat(10)
    # test_one_hot = make_one_hot(args.test_batch_size, test_labels)
    # z_test = torch.cat((z_test, test_one_hot), dim=1)
    # test_images = G(z_test)



    # save_image(denorm(test_images.data), './samples/test_{}_epoch.png'.format(epoch), nrow=num_test_each)
    save_image(denorm(images.data), './samples/fake_{}_epoch.png'.format(epoch), nrow=8)
    save_image(images.data, './samples/fake_raw_{}_epoch.png'.format(epoch), nrow=8)
    save_image(denorm(data.data), './samples/real_image.png', nrow=8)






for epoch in range(1, args.epochs+1):
    train(epoch)

torch.save(D.state_dict(), './models/d_model.pth')
torch.save(G.state_dict(), './models/g_model.pth')
