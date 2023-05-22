import torch
import numpy as np
from torchvision.transforms import transforms
from almost_fully_convolutional import FullyConvolutionLogistic
from analytic_dataset import AnalyticDataset

transform = transforms.Compose(
    [
        transforms.Lambda(
            lambd=lambda x: x.astype(np.float32)
        ),
        transforms.Lambda(
            lambd=lambda x: torch.from_numpy(x)
        ),
    ]
)

ckpt = torch.load('best.ckpt')
model = FullyConvolutionLogistic(100)

model.load_state_dict(ckpt['model_state_dict'])

dataset = AnalyticDataset(1000, transform=transform)

i, o = dataset[0]
torch.stack((model(i.unsqueeze(0)).squeeze(0), o, o - model(i.unsqueeze(0)).squeeze(0))).T

import matplotlib.pyplot as plt

# y = o.detach().numpy()
x = np.arange(0, 100)
# y_hat = model(i.unsqueeze(0)).squeeze(0).detach().numpy()

y_hat = [model(i.unsqueeze(0)).squeeze(0).detach().numpy() for i in [x for x,y in dataset]]
y_hat = np.array(y_hat).mean(axis=0)

plt.scatter(x, np.log(np.abs(y_hat)))