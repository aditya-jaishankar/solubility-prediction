#%% [markdown]

# In this project, we use the solubility data from before and run CNN on images
# generated from rdkit.  These images are also tagged as being hydrophobic and
# hydrophilic. We then see if the CNN can correctly predict whether the structure
# is hydrophilic or hydrophobic based on the the drawing of the structure. 
#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
device = torch.device('cuda:0')

import os
dirpath = os.path.dirname(os.path.realpath(__file__))

from tqdm import tqdm


#%%


images_arr = np.load(dirpath + '/im_tensor.npy')
labels_arr = np.load(dirpath + '/category_tensor.npy')

images_train, images_test, labels_train, labels_test = train_test_split(
    images_arr, labels_arr, test_size=0.2, random_state=25)

# torch requires images in the format (channels, height, width)
images_train = np.transpose(images_train, axes=(0, 3, 1, 2))
images_test = np.transpose(images_test, axes=(0, 3, 1, 2))


#%%
class imagesDataset(Dataset):

    def __init__(self, images, categories):

        super(imagesDataset, self).__init__()

        self.images = images
        self.categories = categories

    def __len__(self):
        # len() specifies the upper bound for the index of the dataset
        return len(self.categories)

    def __getitem__(self, index):
        # The generator executes the getitem() method to generate a sample
        return self.images[index], self.categories[index]

#%%

train_set = imagesDataset(images_train, labels_train)
test_set = imagesDataset(images_test, labels_test)

batch_size = 25
train_generator = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_generator = DataLoader(test_set, batch_size=batch_size, shuffle=True)

#%%
# Next we define the CNN we want to use

class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(4, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 64, (3, 3))

        self.fc1 = nn.Linear(104*104*64, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)

        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.logsoftmax = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)

        x = x.view(-1, 104*104*64)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        x = self.logsoftmax(x)

        return x
    
    # def length(self, x):
    #     linearized = x.view(4, 1, -1)
    #     return linearized.shape[0]

#%%
NUM_EPOCHS = 50
cnn = CNN().to(device)

objective = nn.NLLLoss()
optimizer = optim.SGD(cnn.parameters(), lr=.0001)

#%%
total_loss = torch.zeros(NUM_EPOCHS)
for i in range(NUM_EPOCHS):
    for inp, labels in tqdm(train_generator):
        inp, label = inp.float().to(device), labels.float().to(device)
        out = cnn(inp)

        optimizer.zero_grad()
        # The reshaping is because pytorch needs these dimensions for 
        # NLLLoss to work properly. 
        loss = objective(out.squeeze(), label.long().view(-1))
        loss.backward()
        optimizer.step()
        total_loss[i] += loss.item()/(.8*len(train_generator)*batch_size)
    print('Loss for Epoch', i, '=', total_loss[i])

plt.plot(range(total_loss.shape[0]), total_loss)
f = dirpath + 'cnn_saved.pt'
torch.save(cnn.state_dict(), f)
#%%
f = dirpath + 'cnn_saved.pt'
cnn = CNN()
cnn.load_state_dict(torch.load(f))
cnn.eval()
cnn.to(device)

#%%
#TODO: Calculate a validation score
#TODO: Go back and rewrite the RNN and LSTM codes to make more complex
        # and then train on gpu to see if my predictions improve. I was
        # getting put off a bit by the long training times to iterate, but 
        # this seems much faster and a lot more fun
#TODO: Try the Adam optimizer. Is that going to be any fun?
#TODO: Do I need to apply a composed transpose on the images because there 
        # is so much white

def validation():
    errors = 0
    for inp, label in tqdm(test_generator):
        preds = torch.argmax(cnn(inp.float().to(device)), dim=1)
        labels = label.to(device).view(-1)
        errors += torch.sum((preds==labels)*1).item()
    return errors*100/(len(test_generator)*batch_size)
    
#%%
accuracy = validation()
print('accuracy:', accuracy, '%')
#%%
