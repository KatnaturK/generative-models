import random
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# HYPERPARAMETERS
features = 16
epochs = 200
batch_size = 128
eta = 0.0002
device = 'cuda'

class VAE(nn.Module):
  def __init__(self):
    super(VAE, self).__init__()

    self.conv0 = nn.Conv2d(1, 32, kernel_size = 3, stride = 2, padding = 1)
    self.conv0_drop = nn.Dropout2d(0.25)
    self.conv1 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
    self.conv1_drop = nn.Dropout2d(0.25)
    self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
    self.conv2_drop = nn.Dropout2d(0.25)
    self.conv3 = nn.Conv2d(128, 256, kernel_size = 3, stride = 2, padding = 1)
    self.conv3_drop = nn.Dropout2d(0.25)
    self.fc = nn.Linear(12544, features*2)
    self.fc1 = nn.Linear(16, 256*7*7)
    self.trans_conv1 = nn.ConvTranspose2d(256, 128, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
    self.trans_conv2 = nn.ConvTranspose2d(128, 64, kernel_size = 3, stride = 1, padding = 1)
    self.trans_conv3 = nn.ConvTranspose2d(64, 32, kernel_size = 3, stride = 1, padding = 1)
    self.trans_conv4 = nn.ConvTranspose2d(32, 1, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)

  def reparameterize(self, mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    sample = mu + (eps * std)
    return sample

  def generate(self, sample):
    x = self.fc1(sample)
    x = x.view(-1, 256, 7, 7)
    x = F.relu(self.trans_conv1(x))
    x = F.relu(self.trans_conv2(x))
    x = F.relu(self.trans_conv3(x))
    x = self.trans_conv4(x)
    generated = torch.sigmoid(x)
    return generated
 
  def forward(self, x):
    x = x.view(-1, 1, 28, 28)
    x = F.leaky_relu(self.conv0(x), 0.2)
    x = self.conv0_drop(x)
    x = F.leaky_relu(self.conv1(x), 0.2)
    x = self.conv1_drop(x)
    x = F.leaky_relu(self.conv2(x), 0.2)
    x = self.conv2_drop(x)
    x = F.leaky_relu(self.conv3(x), 0.2)
    x = self.conv3_drop(x)
    x = x.view(-1, 12544)
    x = self.fc(x)

    x = x.view(-1, 2, features)

    mu = x[:, 0, :]
    log_var = x[:, 1, :]

    z = self.reparameterize(mu, log_var)

    x = self.fc1(z)
    x = x.view(-1, 256, 7, 7)
    x = F.relu(self.trans_conv1(x))
    x = F.relu(self.trans_conv2(x))
    x = F.relu(self.trans_conv3(x))
    x = self.trans_conv4(x)
    reconstruction = torch.sigmoid(x)
    return reconstruction, mu, log_var


model = VAE()
print(model)
model = model.to(device)
model.float()


transform = transforms.Compose([
  transforms.ToTensor(),
])

train_data = datasets.MNIST(
  root='../input/data',
  train=True,
  download=True,
  transform=transform
)
val_data = datasets.MNIST(
  root='../input/data',
  train=False,
  download=True,
  transform=transform
)

train_loader = torch.utils.data.DataLoader(
  train_data,
  batch_size=batch_size,
  shuffle=True
)
val_loader = torch.utils.data.DataLoader(
  val_data,
  batch_size=batch_size,
  shuffle=False
)

normal_samples = torch.randn(36, features).to(device)

def final_loss(bce_loss, mu, logvar):
  BCE = bce_loss 
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  return BCE + KLD

optimizer = optim.Adam(model.parameters(), lr=eta)
criterion = nn.BCELoss(reduction='sum')

def fit(model, dataloader):
  model.train()
  running_loss = 0.0
  for i, data in enumerate(dataloader):
    data, _ = data
    data = data.to(device)
    optimizer.zero_grad()
    reconstruction, mu, logvar = model(data)
    bce_loss = criterion(reconstruction, data)
    loss = final_loss(bce_loss, mu, logvar)
    running_loss += loss.item()
    loss.backward()
    optimizer.step()
  train_loss = running_loss/len(dataloader.dataset)
  return train_loss

def validate(model, dataloader, samples):
  model.eval()
  running_loss = 0.0
  with torch.no_grad():
    for i, data in enumerate(dataloader):
      data, _ = data
      data = data.to(device)
      reconstruction, mu, logvar = model(data)
      reconstruction = reconstruction.to(device)
      bce_loss = criterion(reconstruction, data)
      loss = final_loss(bce_loss, mu, logvar)
      running_loss += loss.item()


      if i == int(len(val_data)/dataloader.batch_size) - 1:
        num_rows = 8
        samples.append(torch.cat((data.view(batch_size, 1, 28, 28)[:num_rows], 
                           reconstruction.view(batch_size, 1, 28, 28)[:num_rows])))

  val_loss = running_loss/len(dataloader.dataset)
  return val_loss

def view_samples(samples, epoch):
  samples = samples.to('cpu')
  fig, axes = plt.subplots(figsize=(4,4), nrows=6, ncols=6, sharex=True, sharey=True)
  for ax, img in zip(axes.flatten(), samples):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')
  plt.savefig('graphs/CNN VAE Epoch ' + str(epoch) + '.png')

train_loss = []
val_loss = []
samples = []

for epoch in range(epochs):
    train_epoch_loss = fit(model, train_loader)
    val_epoch_loss = validate(model, val_loader, samples)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    print('Epoch [{:3d}/{:3d}] | Train loss: {:5.2f} | Val loss: {:5.2f}'.format(
                    epoch+1, epochs, train_epoch_loss, val_epoch_loss))
    if epoch % 50 == 0:
      model.eval()
      generated_images = model.generate(normal_samples)
      generated_images = generated_images.view(36, 1, 28, 28)
      view_samples(generated_images, epoch+1)

fig, ax = plt.subplots()
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='Val loss')
plt.title("Training Losses")
plt.legend()