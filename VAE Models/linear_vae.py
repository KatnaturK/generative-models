import random
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.optim as optim
import matplotlib.pyplot as plt

# HYPERPARAMETERS
features = 16
epochs = 200
batch_size = 128
eta = 0.0002
device = 'cuda'

class LinearVAE(nn.Module):
  def __init__(self):
    super(LinearVAE, self).__init__()

    self.encoder = nn.Sequential(
        nn.Linear(in_features=784, out_features=512),
        nn.ReLU(),
        nn.Linear(in_features=512, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=features*2)
    )

    self.decoder = nn.Sequential(
      nn.Linear(in_features=features, out_features=128),
      nn.ReLU(),
      nn.Linear(in_features=128, out_features=512),
      nn.ReLU(),
      nn.Linear(in_features=512, out_features=784),
      nn.Sigmoid()
    )

  def generate(self, sample):
    generated = self.decoder(sample)
    return generated
   
  def forward(self, x):
    x = self.encoder(x)
    x = x.view(-1, 2, features)
    mu = x[:, 0, :]
    log_var = x[:, 1, :]
    z = self.estimate(mu, log_var)

    reconstruction = self.decoder(z)
    return reconstruction, mu, log_var

  def estimate(self, mu, log_var):
    std = torch.exp(0.5*log_var)
    eps = torch.randn_like(std)
    sample = mu + (eps * std)
    return sample


model = LinearVAE()
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
    data = data.view(data.size(0), -1).to(device)
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
      data = data.view(data.size(0), -1).to(device)
      reconstruction, mu, logvar = model(data)
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
  plt.savefig('graphs/Linear VAE Epoch ' + str(epoch) + '.png')

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
    if epoch % 5 == 0:
      model.eval()
      generated_images = model.generate(normal_samples)
      generated_images = generated_images.view(36, 1, 28, 28)
      view_samples(generated_images, epoch)

fig, ax = plt.subplots()
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='Val loss')
plt.title("Training Losses")
plt.legend()