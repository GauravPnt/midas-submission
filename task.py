"""
Checkpoints of the model are going to be saved in root_dir/models
Provide root_dir while running the application as cmd line argument
Place the dataset zips from the repo in the root_dir
"""


import matplotlib.pyplot as plt
from torch.optim import AdamW
import tqdm
import copy
from torch import nn
from PIL import Image
import numpy as np
from cv2 import cv2
import torchvision
import torch.utils.data as data
from torchvision import transforms
from zipfile import ZipFile
import torch
import os
import sys
root_dir = sys.argv[1]
if os.path.isdir(root_dir) == False:
    os.makedirs(root_dir)
os.chdir(root_dir)


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)

"""
Function for unzipping the file
Function for getting the dataset from a folder path
"""


def unzip(file_name):
    with ZipFile(file_name, 'r') as zip:
        # printing all the contents of the zip file
        # zip.printdir()
        # extracting all the files
        print('Extracting all the files now...')
        zip.extractall()
        print('Done!')


def get_data_set(folder_path):
    transform = [
        # transforms.CenterCrop(256),
        # Crop(28),
        # transforms.Resize(256),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
    data = torchvision.datasets.ImageFolder(
        root=folder_path, transform=transforms.Compose(transform))
    return data


"""Objective of this class is to crop the high resolution images and invert them to make them in line with the MNIST dataset"""


class Crop(object):

    def __init__(self, size):
        self.size = size
        assert isinstance(size, int)

    def make_square(self, im, fill_color=(0)):
        x, y = im.size
        ratio = x / y
        sz = self.size
        # Ensure that the image has some padding inside square
        sz -= 4
        if x > y:
            y = sz / ratio
            x = sz
        else:
            y = sz
            x = sz * ratio
        y, x = int(y), int(x)
        # print(x, y)
        im = im.resize((x, y))
        new_im = Image.new('L', (self.size, self.size), fill_color)
        new_im.paste(im, (int((self.size - x) / 2), int((self.size - y) / 2)))
        # print(new_im.size)
        return new_im

    def crop(self, sample):
        img = np.array(sample)
        img = img[:, :, ::-1].copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

        # Find the max-area contour
        # print(sample)
        cnts = cv2.findContours(
            threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnt = sorted(cnts, key=cv2.contourArea)[-1]

        # Crop and save it
        x, y, w, h = cv2.boundingRect(cnt)
        gray = cv2.bitwise_not(gray)
        dst = gray[y:y+h, x:x+w]
        # print(dst.shape[:2])
        im_pil = Image.fromarray(dst)
        res = self.make_square(im_pil)
        # display(res)
        return res

    def crop_all(self, rootdir):
        print(f'Cropping all images now to {self.size} x {self.size} ...')
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                path = os.path.join(subdir, file)
                img = Image.open(path)
                img = self.crop(img)
                root_dir = os.path.join(
                    rootdir + '_modified', path[path.find('/') + 1: path.rfind('/')])
                # print(root_dir)
                if os.path.isdir(root_dir) == False:
                    os.makedirs(root_dir)
                save_path = os.path.join(
                    rootdir + '_modified', path[path.find('/') + 1:])
                # print(save_path)
                img.save(save_path)
        print(f'cropped images saved to {rootdir}_modified')
        return rootdir + '_modified'


unzip('trainPart1.zip')


"""
Apply pre-processing on the provided dataset -

1.   Crop the images in the dataset to size 28 x 28 by applying rectangular bounding box on the images
2.   Invert the color of the image so as to match with the dataset of mnist
3.   Finally save these in a separate folder and provide its path
"""

cp = Crop(28)
cropped_path_62 = cp.crop_all('train')

"""
This zip has been created manually from the trainining data and contains the data points for digits 0 - 9 only
"""

unzip('train_digits.zip')

cp = Crop(28)
cropped_path_10 = cp.crop_all('train_digits')

unzip('mnistTask3.zip')

"""
Define the CNN architecture
Final layers in the model can vary according to the number of characters in the dataset
"""


class CNN(nn.Module):
    def __init__(self, fc):
        super().__init__()
        self.cnn_net = nn.Sequential(
            nn.Conv2d(1, 64, 3),  # 28 x 28 -> 26 x 26
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3),  # 26 x 26 -> 24 x 24
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 24 x 24 -> 12 x 12
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3),  # 12 x 12 -> 10 x 10
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3),  # 10 x 10 -> 8 x 8
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8 x 8 -> 4 x 4
            nn.Flatten(1),
            nn.BatchNorm1d(128 * 4 * 4),
            nn.Dropout(p=0.25),
            nn.Linear(128 * 4 * 4, fc[0]),
            nn.ReLU(),
            nn.BatchNorm1d(fc[0]),
            nn.Dropout(p=0.25),
            nn.Linear(fc[0], fc[1]),
            nn.ReLU(),
            nn.BatchNorm1d(fc[1]),
            nn.Dropout(p=0.25),
            nn.Linear(fc[1], fc[2])
        )

    def forward(self, x):
        return self.cnn_net(x)


"""Define the final fc layers for 62 char and 10 char model"""

FC_62 = (1024, 512, 62)
FC_10 = (512, 128, 10)

"""
Define a model class with the following objectives -

load - load the pretrained model from memory
train - train the model from the given trainloader
test - test the model from the given testloader
test_mnist - test the model on MNIST test set
save - save the checkpoint
train_validate - do train, validate split and run the code for specified number of epochs and provide accuracy on the test and validate dataset
"""


class Model:
    def __init__(self, epochs=50, fc=FC_62):
        self.epochs = epochs
        self.model = CNN(fc)
        self.model.to(device)
        self.num_epochs = epochs
        self.epochs = 0
        self.loss = 0
        self.optimizer = AdamW(params=self.model.parameters())
        self.loss_fn = nn.CrossEntropyLoss()
        self.transform2 = [
            # transforms.CenterCrop(256),
            # Crop(28),
            # transforms.Resize(256),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]

    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epochs = checkpoint['epoch']
        self.loss = checkpoint['loss']
        print(f'\nmodel loaded from path : {path}')

    def save(self, epoch, model, optimizer, loss, path):
        save_path = root_dir + '/models/'
        if os.path.isdir(save_path) == False:
            os.makedirs(save_path)
        path = save_path + path
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)
        print(f'\nsaved model to path : {path}')

    def test(self, testloader, progress, type='validation'):
        print(f'Starting testing on {type} dataset')
        print('-------------------------------')
        correct, total = 0, 0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, targets = data
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = self.model(inputs)

                _, predicted = torch.max(outputs.data, 1)
                # print(predicted)
                # print(targets)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                progress.update(self.batch_size)

            print(
                f'\nAccuracy on {type} dataset : {correct} / {total} = {100.0 * correct / total}')
            print('--------------------------------')

            return 100.0 * correct / total

    def train(self, trainloader, epoch, progress):
        print(f'\nStarting epoch {epoch+1}')
        current_loss = 0.0

        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            self.optimizer.step()
            current_loss += loss.item()
            progress.update(self.batch_size)

        print(f'\nloss at epoch {epoch + 1} : {current_loss}')
        return current_loss

    def train_validate(self, name, mnist=False, batch_size=64, validation_split=0, save_name=None):
        self.batch_size = batch_size

        if save_name is None:
            save_name = name

        progress = None
        np.random.seed(42)

        epochs_plot = []
        accuracy_plot = []
        loss_plot = []

        for epoch in range(0, self.num_epochs):
            if mnist:
                self.transform1 = [
                    transforms.RandomRotation(degrees=10),
                ]
                train_data = torchvision.datasets.MNIST(
                    'mnist', download=True, transform=transforms.Compose(self.transform1 + self.transform2))
                trainloader = torch.utils.data.DataLoader(
                    train_data, batch_size=self.batch_size, num_workers=2)
                dataset_size = len(trainloader.dataset)
            else:
                data = get_data_set(name)
                dataset_size = len(data)
                ids = list(range(dataset_size))
                split = int(np.floor(validation_split * dataset_size))
                np.random.shuffle(ids)
                train_ids, val_ids = ids[split:], ids[:split]

                train_subsampler = torch.utils.data.SubsetRandomSampler(
                    train_ids)
                test_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

                trainloader = torch.utils.data.DataLoader(
                    data,
                    batch_size=batch_size,
                    sampler=train_subsampler,
                    num_workers=2
                )
                testloader = torch.utils.data.DataLoader(
                    data,
                    batch_size=batch_size,
                    sampler=test_subsampler,
                    num_workers=2
                )
            if progress is None:
                progress = tqdm.tqdm(total=(
                    2 + validation_split)*dataset_size*self.num_epochs, position=0, leave=True)
            current_loss = self.train(trainloader, epoch, progress)
            accuracy = self.test(trainloader, progress, 'train')
            if validation_split:
                self.test(testloader, progress, 'validation')
            epochs_plot.append(epoch)
            accuracy_plot.append(accuracy)
            loss_plot.append(current_loss)
            self.save(epoch, self.model, self.optimizer,
                      current_loss, f'{save_name}-{epoch}.pth')
        return epochs_plot, accuracy_plot, loss_plot

    def test_mnist(self):
        test_data = torchvision.datasets.MNIST(
            'mnist', False, download=True, transform=transforms.Compose(self.transform2))
        testloader = torch.utils.data.DataLoader(
            test_data, batch_size=self.batch_size, num_workers=2)
        progress = tqdm.tqdm(total=len(testloader.dataset),
                             position=0, leave=True)
        self.test(testloader, progress, 'test')


"""Function to plot the loss and accuracy plot corresponding to the epochs"""


def plot(epochs_plot, loss_plot, accuracy_plot):
    plt.plot(epochs_plot, accuracy_plot, label="Accuracy plot")
    plt.xlabel('Epochs')

    plt.legend()
    plt.show()

    plt.plot(epochs_plot, loss_plot, label="Loss plot")

    plt.legend()
    plt.show()


"""Print out the architecture for 62 char model"""

high_res = Model()
print('Parameters for 62 char model')
print(high_res.model)

"""Print out the architecture for 10 char model"""

mnist_pre_trained = Model(fc=FC_10)
print('Parameters for 10 char model')
print(mnist_pre_trained.model)

"""Train the model on the 62 characters dataset provided with 9:1 train - validation split so as to check how good the cnn architecture is"""

high_res = Model(100)
epochs_plot, accuracy_plot, loss_plot = high_res.train_validate(
    cropped_path_62, validation_split=0.1)
plot(epochs_plot, loss_plot, accuracy_plot)

"""Train with the complete training data since we have now determined our CNN architecture"""

high_res = Model(100)
epochs_plot, accuracy_plot, loss_plot = high_res.train_validate(
    cropped_path_62)
plot(epochs_plot, loss_plot, accuracy_plot)

"""Tweak the final fc layers of the CNN architecture and train on the provided datasets but now with the dataset containing only 0-9 characters"""

mnist_pre_trained = Model(fc=FC_10)
epochs_plot, accuracy_plot, loss_plot = mnist_pre_trained.train_validate(
    cropped_path_10, validation_split=0)
plot(epochs_plot, loss_plot, accuracy_plot)

"""Retrain this pre trained model with the mnist data set"""

epochs_plot, accuracy_plot, loss_plot = mnist_pre_trained.train_validate(
    name='train_digits_retrain', mnist=True, batch_size=256, validation_split=0, save_name='train_digits_retrain')
plot(epochs_plot, loss_plot, accuracy_plot)

"""Test the accuracy of the  retrained network on the MNIST test dataset"""

mnist_pre_trained.test_mnist()

"""Train the same CNN architecture but this time with random initialization on the MNIST dataset"""

mnist_random = Model(fc=FC_10)
epochs_plot, accuracy_plot, loss_plot = mnist_random.train_validate(
    name='mnist_vanilla', mnist=True, batch_size=256, save_name='mnist_vanilla')
plot(epochs_plot, loss_plot, accuracy_plot)

"""Test the accuracy of the random initialized CNN on the MNIST test dataset"""

mnist_random.test_mnist()

"""Train the pre trained network on the shuffled MNIST dataset"""

destroy_mnist_pretrained = Model(fc=FC_10)
destroy_mnist_pretrained.load(
    '/content/drive/MyDrive/Colab Notebooks/models/train_digits-49.pth')
epochs_plot, accuracy_plot, loss_plot = destroy_mnist_pretrained.train_validate(
    name='mnistTask', mnist=False, batch_size=256, validation_split=0, save_name='destroy_pretrained_mnist')
plot(epochs_plot, loss_plot, accuracy_plot)

"""Test accuracy of the pre trained network on the MNIST test dataset"""

destroy_mnist_pretrained.test_mnist()

"""Train a randomly initialized CNN on shuffled MNIST dataset"""

destroy_mnist_random = Model(fc=FC_10)
epochs_plot, accuracy_plot, loss_plot = destroy_mnist_random.train_validate(
    name='mnistTask', mnist=False, batch_size=256, validation_split=0, save_name='destroy_random_mnist')
plot(epochs_plot, loss_plot, accuracy_plot)

"""Check accuracy of the randomly initialized network on the MNIST test dataset"""

destroy_mnist_random.test_mnist()
