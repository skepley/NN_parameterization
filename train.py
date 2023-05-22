from almost_fully_convolutional import DenseLogistic
import torch
from torch import nn
from analytic_dataset import AnalyticDataset
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np

DEVICE_ID = 'mps'
EPOCHS = 100
LR = .0005
DATASET_SIZE = 100
TRAINING_DATA_SIZE = int(0.9 * DATASET_SIZE)
N = 100


def dataSampler(dataSetSize):
    return np.random.uniform(low=-1, high=1, size=dataSetSize)


dataset = AnalyticDataset(DATASET_SIZE, dataSampler, 1 + N)
lengths = (TRAINING_DATA_SIZE, DATASET_SIZE - TRAINING_DATA_SIZE)
trainSet, valSet = random_split(dataset, lengths)
trainloader = DataLoader(trainSet, batch_size=100)
valloader = DataLoader(valSet, batch_size=100)

# model = FullyConvolutionLogistic(100)
model = DenseLogistic('./model_configs/dense_logistic/base.yml')

model.to(DEVICE_ID)
optim = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.L1Loss()
# loss_fn = nn.L1Loss(reduction='none')  # this function does not sum over the batch or even the sequences themselves i.e. its just the pointwise absolute value and returns something of the same size as the input
# weights = torch.exp(-torch.arange(0, 100)).to(DEVICE_ID)

best_loss = np.Inf
model.train()

if __name__ == '__main__':
    for epoch in range(EPOCHS):
        total_loss = 0
        for in_batch, out_batch in tqdm(
                trainloader):  # in_batch is a batch of initial values, out_batch is a batch of NN outputs
            in_batch = in_batch.to(DEVICE_ID)
            out_batch = out_batch.type(dtype=torch.float32)  # coerce float64 data into float32 is required for mps
            out_batch = out_batch.to(DEVICE_ID)
            optim.zero_grad()  # zero out gradient information from the previous batch (but keeps the momentum and higher order data)

            pred = model(in_batch)
            # loss = (loss_fn(pred, out_batch) * weights).sum()
            loss = loss_fn(pred,
                           out_batch)  # NOTE: By default this function returns the mean of the loss function evaluated over the entire batch i.e. this is ALWAYS a scalar
            loss.backward()  # back propagation
            optim.step()  # update NN parameters
            total_loss += loss.data
        print(f'Epoch {epoch} | Loss {total_loss}')
        torch.save(
            {
                'epoch': epoch,
                'loss': total_loss,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict()
            },
            'latest.ckpt',
        )
        if total_loss < best_loss:
            torch.save(
                {
                    'epoch': epoch,
                    'loss': total_loss,
                    'model_state_dict': model.state_dict(),
                    'optim_state_dict': optim.state_dict()
                },
                'best.ckpt',
            )
