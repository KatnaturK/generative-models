import numpy as np
import torch
from torch import nn
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt

batch_size = 128
eta = 0.0002
epochs = 200
DROPOUT_PROB = 0.4

device = 'cuda'


class Discriminator(nn.Module):

    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        input_size = 28 * 28
        output_size = 1

        self.out = nn.Sequential(
            nn.Linear(input_size, 4 * hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT_PROB),
            nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT_PROB),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(DROPOUT_PROB),
            nn.Linear(hidden_dim, output_size)
        )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        return self.out(x)


class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim):
        super(Generator, self).__init__()
        output_size = 28 * 28

        self.out = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(2 * hidden_dim, 4 * hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(4 * hidden_dim, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.out(x)


D = Discriminator(128)
G = Generator(100, 128)
print(D)
print(G)

D = D.to(device)
G = G.to(device)

D = D.float()
G = G.float()


def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(.5, .5)
         ])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)


data = mnist_data()

data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

num_batches = len(data_loader)
print("Batch: {} with batch size: {}".format(num_batches, batch_size))


def real_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.ones(batch_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


def calculate_real_acc(D_out):
    labels = torch.ones(D_out.size()[0]).to(device)
    correct_pred = ((D_out.squeeze() >= 0.5) == labels).float().sum()
    return correct_pred.item()


def calculate_fake_acc(D_out):
    labels = torch.zeros(D_out.size()[0]).to(device)
    correct_pred = ((D_out.squeeze() >= 0.5) == labels).float().sum()
    return correct_pred.item()


def sample_input(batch_size=-1):
    if batch_size != -1:
        return torch.normal(mean=torch.zeros((batch_size, 100)),
                            std=torch.ones((batch_size, 100)))
    else:
        return torch.normal(mean=torch.zeros(100),
                            std=torch.ones(100))


sample_size = 36
fixed_z = sample_input(sample_size).to(device)

d_optimizer = optim.Adam(D.parameters(), eta)
g_optimizer = optim.Adam(G.parameters(), eta)


def view_samples(samples, epoch):
    samples = samples.to('cpu')
    fig, axes = plt.subplots(figsize=(4, 4), nrows=6, ncols=samples.size()[0] // 6, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
    plt.savefig('graphs/Linear GAN Epoch ' + str(epoch) + '.png')


samples = []
losses = []
real_acc, fake_acc = [], []

print_frequency = 60000 // batch_size + 1

D.train()
G.train()
for epoch in range(epochs):
    real_count, fake_count = 0, 0
    for batch_i, (real_images, _) in enumerate(data_loader):

        batch_size = real_images.size(0)
        real_images = real_images.to(device)

        z = sample_input(batch_size).to(device)

        D.eval()
        G.eval()
        fake_images = G(z)
        real_count += calculate_real_acc(D(real_images))
        fake_count_in_batch = calculate_fake_acc(D(fake_images))
        fake_count += fake_count_in_batch
        G.train()
        D.train()

        d_optimizer.zero_grad()

        D_real = D(real_images)
        d_real_loss = real_loss(D_real)

        fake_images = G(z)
        D_fake = D(fake_images)
        d_fake_loss = fake_loss(D_fake)

        d_loss = d_real_loss

        d_loss += d_fake_loss

        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()

        z = sample_input(batch_size).to(device)
        fake_images = G(z)

        D_fake = D(fake_images)
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

    G.eval()
    samples_z = G(fixed_z)
    samples.append(samples_z)
    if epoch % 50 == 0:
        view_samples(samples_z, epoch)
    G.train()

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
