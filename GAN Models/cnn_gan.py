import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

batch_size = 128
eta = 0.0002
generator_features = 100
generator_hidden_shape = 128
discriminator_hidden_shape = 128
epochs = 200
DROPOUT_PROB = 0.4

device = 'cuda'


class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 256 * 7 * 7)
        self.trans_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.trans_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.trans_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.trans_conv4 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 7, 7)
        x = F.relu(self.trans_conv1(x))
        x = F.relu(self.trans_conv2(x))
        x = F.relu(self.trans_conv3(x))
        x = self.trans_conv4(x)
        x = torch.tanh(x)
        return x


class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.convolution0 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.convolution0_drop = nn.Dropout2d(0.25)
        self.convolution1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.convolution1_drop = nn.Dropout2d(0.25)
        self.convolution2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.convolution2_drop = nn.Dropout2d(0.25)
        self.convolution3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.convolution3_drop = nn.Dropout2d(0.25)
        self.fc = nn.Linear(12544, 1)

    def number_of_flat_features(self, x):
        values = x.size()[1:]
        number_of_features = 1
        for value in values:
            number_of_features *= value

        return number_of_features

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.leaky_relu(self.convolution0(x), 0.2)
        x = self.convolution0_drop(x)
        x = F.leaky_relu(self.convolution1(x), 0.2)
        x = self.convolution1_drop(x)
        x = F.leaky_relu(self.convolution2(x), 0.2)
        x = self.convolution2_drop(x)
        x = F.leaky_relu(self.convolution3(x), 0.2)
        x = self.convolution3_drop(x)
        x = x.view(-1, self.number_of_flat_features(x))
        x = self.fc(x)
        return x


Discrim = Discriminator()
Gen = Generator()

print(Discrim)
print(Gen)

Discrim = Discrim.to(device)
Gen = Gen.to(device)

Discrim = Discrim.float()
Gen = Gen.float()


def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(.5, .5)
         ])
    output_directory = './dataset'
    return datasets.MNIST(root=output_directory, train=True, transform=compose, download=True)


data = mnist_data()

loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

num_batches = len(loader)
print("Batch: {} with batch size: {}".format(num_batches, batch_size))


def real_loss(discrim_out):
    batch_size = discrim_out.size(0)
    labels = torch.ones(batch_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(discrim_out.squeeze(), labels)
    return loss


def fake_loss(discrim_out):
    batch_size = discrim_out.size(0)
    labels = torch.zeros(batch_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(discrim_out.squeeze(), labels)
    return loss


def calculate_fake_acc(discrim_out):
    labels = torch.zeros(discrim_out.size()[0]).to(device)
    prediction = ((discrim_out.squeeze() >= 0.5) == labels).float().sum()
    return prediction.item()


def calculate_real_acc(discrim_out):
    labels = torch.ones(discrim_out.size()[0]).to(device)
    prediction = ((discrim_out.squeeze() >= 0.5) == labels).float().sum()
    return prediction.item()


def sample_input(batch_length=-1):
    if batch_length != -1:
        return torch.normal(mean=torch.zeros((batch_length, generator_features)),
                            std=torch.ones((batch_length, generator_features)))
    else:
        return torch.normal(mean=torch.zeros(generator_features),
                            std=torch.ones(generator_features))


sample_size = 36
fixed_z = sample_input(sample_size).to(device)

d_optimizer = optim.Adam(Discrim.parameters(), eta)
g_optimizer = optim.Adam(Gen.parameters(), eta)

samples = []
losses = []
real_acc, fake_acc = [], []

print_frequency = 60000 // batch_size + 1

Discrim.train()
Gen.train()
for epoch in range(epochs):
    real_count, fake_count = 0, 0
    for batch_i, (real_images, _) in enumerate(loader):

        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        z = sample_input(batch_size).to(device)

        Discrim.eval()
        Gen.eval()
        fake_images = Gen(z)
        real_count += calculate_real_acc(Discrim(real_images))
        fake_count_in_batch = calculate_fake_acc(Discrim(fake_images))
        fake_count += fake_count_in_batch
        Gen.train()
        Discrim.train()

        d_optimizer.zero_grad()

        D_real = Discrim(real_images)
        d_real_loss = real_loss(D_real)

        fake_images = Gen(z)
        D_fake = Discrim(fake_images)
        d_fake_loss = fake_loss(D_fake)

        d_loss = d_real_loss

        d_loss += d_fake_loss

        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()

        z = sample_input(batch_size).to(device)
        fake_images = Gen(z)

        D_fake = Discrim(fake_images)
        g_loss = real_loss(D_fake)

        g_loss.backward()
        g_optimizer.step()

        if batch_i % print_frequency == 0:
            print('Epoch [{:3d}/{:3d}]'.format(epoch + 1, epochs))
            print("\td_real_loss: {:5.2f} | d_fake_loss: {:5.2f} | g_loss: {:5.2f}".format(d_real_loss.item(),
                                                                                           d_fake_loss.item(),
                                                                                           g_loss.item()))
            print(
                "\td_fake_acc: {:5.2f} | d_real_acc: {:5.2f}".format(fake_count / batch_size, real_count / batch_size))

    losses.append((d_real_loss.item() + d_fake_loss.item(), g_loss.item()))
    real_acc.append(real_count / 60000)
    fake_acc.append(fake_count / 60000)

    Gen.eval()
    samples_z = Gen(fixed_z)
    samples.append(samples_z)
    Gen.train()

fig, (ax1, ax2) = plt.subplots(figsize=(10, 4), nrows=1, ncols=2)
losses = np.array(losses)
ax1.plot(losses.T[0], label='Discriminator Loss')
ax1.plot(losses.T[1], label='Generator Loss')
ax1.set_title("Training Stats")
ax2.plot(fake_acc, label='Fake Accuracies')
ax2.plot(real_acc, label='Real Accuracies')
ax2.set_title("Discriminator Accuracy")
ax1.legend()
ax2.legend()


def view_samples(samples, epoch):
    samples = samples.to('cpu')
    fig, axes = plt.subplots(figsize=(4, 4), nrows=6, ncols=samples.size()[0] // 6, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
    plt.savefig('graphs/CNN GAN Epoch ' + str(epoch) + '.png')


Gen.eval()
sample = Gen(fixed_z)
view_samples(sample)
