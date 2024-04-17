import matplotlib.pyplot as plt
import torch
import torch.nn as nn # all the relevant building blocks
import torch.nn.functional as F # functional interfaces for many operations
from torch.utils.data import Dataset, DataLoader # abstract primitives for handling data in pytorch
from torchvision import transforms

from torchvision.models import resnet18, ResNet18_Weights

import pickle
import numpy as np

if torch.cuda.is_available():
    print("GPU is available")
    device = torch.device("cuda")
else:
    print("GPU is not available, using CPU instead")
    device = torch.device("cpu")

USE_WANDB = False

if USE_WANDB:
    import wandb
    run = wandb.init(
        project="hpc_tutorial",
        # Track hyperparameters and run metadata
        config={
            "learning_rate": 0.01,
            "epochs": 5,
        },
    )


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def fit(model, train_loader, val_loader, n_epochs, loss_fn, optimizer):
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for epoch in range(n_epochs):
        epoch_loss, val_epoch_loss = 0.0, 0.0
        epoch_acc, val_epoch_acc = 0.0, 0.0

        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()   # reseting gradients

            # Forward pass
            outputs = model(images)

            loss = loss_fn(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += torch.mean((torch.argmax(outputs.detach().cpu(), dim=1) == labels.cpu()).float())

        model.eval()
        with torch.no_grad():  # since we are not going to do backprop
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass only
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_epoch_loss += loss.item()
                val_epoch_acc += torch.mean((torch.argmax(outputs.detach().cpu(), dim=1) == labels.cpu()).float())

        if history['val_accuracy'] and val_epoch_acc > history['val_accuracy'][-1]:
            torch.save(model.state_dict(), f'best_model_epoch_{epoch}.pth')

        history['loss'].append(epoch_loss/len(train_loader))
        history['accuracy'].append(epoch_acc/len(train_loader))

        history['val_loss'].append(val_epoch_loss/len(val_loader))
        history['val_accuracy'].append(val_epoch_acc/len(val_loader))


        print(f"Epoch {epoch + 1}, Loss: {history['loss'][-1]}, Val loss: {history['val_loss'][-1]}")

        if USE_WANDB:
            wandb.log({
                "loss": history['loss'][-1],
                "val_loss": history['val_loss'][-1],
                "accuracy": history['accuracy'][-1],
                "val_accuracy": history['val_accuracy'][-1],
            })

    return history


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images).detach().cpu().numpy()
            preds.extend(list(np.argmax(outputs, axis=1)))
    return np.array(preds)


class CIFARDataset():
    def __init__(self, type):
        if type == 'train':
            for i in range(1, 6):
                data = unpickle(f'./data/cifar-10-batches-py/data_batch_{i}')
                if i == 1:
                    self.data = data
                else:
                    self.data[b'data'] = np.concatenate((self.data[b'data'], data[b'data']))
                    self.data[b'labels'] += data[b'labels']
        elif type == 'test':
            self.data = unpickle('./data/cifar-10-batches-py/test_batch')

        self.transform = transforms.Compose([
                        transforms.ToTensor()
                        ])
        self.labels = np.array(self.data[b'labels'])

    def __len__(self):
        return len(self.data[b'data'])

    def __getitem__(self, idx):
        img = self.data[b'data'][idx]
        img = np.transpose(np.reshape(img, (3, 32, 32)), (1, 2, 0))
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def plot_curves(history):
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])
    plt.title('Accuracy')
    plt.savefig('training_curves.png')


if __name__ == '__main__':
    # Load the dataset
    train_dataset = CIFARDataset('train')
    test_dataset = CIFARDataset('test')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define the model
    # model = torch.hub.load("pytorch/vision", "resnet18", weights="IMAGENET1K_V2")
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    history = fit(model, train_loader, val_loader, 5, criterion, optimizer)
    plot_curves(history)

    preds = predict(model, val_loader)

    print(f'Final accuracy is: {np.mean(preds == test_dataset.labels)}')

