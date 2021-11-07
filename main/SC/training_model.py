import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from main.SC.data_format import SoundDataset
from main.SC.models import Net
from tqdm import tqdm

# set hyperparameters
RANDOM_SEED = 1
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0001
BATCH_SIZE = 20
NUM_EPOCHS = 41
GAMMA = 0.1
LOG_INTERVAL = 20


def load_data():
    csv_path = '/home/chen/Downloads/UrbanSound8K/metadata/UrbanSound8K.csv'
    file_path = '/home/chen/Downloads/UrbanSound8K/audio/'

    train_set = SoundDataset(csv_path, file_path, range(1,10))
    test_set = SoundDataset(csv_path, file_path, [10])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, **kwargs)

    return train_loader, test_loader


def train(num_epoch, train_loader):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay= WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=BATCH_SIZE, gamma=GAMMA)

    for epoch in tqdm(range(1, num_epoch)):
        scheduler.step()
        # Tranining
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            target = target.to(device)
            data = data.requires_grad_()  # set requires_grad to True for training
            output = model(data)
            output = output.permute(1, 0, 2)  # original output dimensions are batchSizex1x10
            loss = F.nll_loss(output[0], target)  # the loss functions expects a batchSizex10 input
            loss.backward()
            optimizer.step()
            if batch_idx % LOG_INTERVAL == 0:  # print training stats
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss))
        # Test
        model.eval()
        correct = 0
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            output = output.permute(1, 0, 2)
            pred = output.max(2)[1]  # get the index of the max log-probability
            correct += pred.eq(target).cpu().sum().item()
        print('\nTest set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    torch.save(model, '/home/chen/PycharmProjects/SoundClassification/model/m5.pth')


if __name__ == '__main__':

    train_loader, test_loader = load_data()
    train(NUM_EPOCHS, train_loader)



