from torch._C import TensorType, dtype
from torch.types import Number
from torch.utils import data
from torch.utils.data.dataset import Dataset
from typing_extensions import runtime
from NeuralNetwork import MNISTModel
import torch
import torchvision
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import torch.nn as nn
from torch import optim
import torch.nn.functional as functional
import matplotlib.pyplot as plt
from typing import Tuple
import os
from CommonLibrary import try_parse_int

class NeuralNetworkDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super().__init__()
        self.data = dataframe

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        numpyImage = self.data.iloc[index,1:].to_numpy()
        label = self.data.iloc[index,0]
        return numpyImage, label

class NeuralNetworkModel(nn.Module):
    """
    Model that defines the structure of a neural network and provides the necessary functions to train and use this neural network
    """
    RunNum = 0
    def __init__(self) -> None:
        """
        The constructor loads data, create a neural network structure
        """
        super(NeuralNetworkModel, self).__init__()
        self.load_data()
        self.create_structure()
        # Check for GPU device
        if torch.cuda.is_available():  
            self.device = torch.device("cuda:0")
            self = self.to(torch.device("cuda:0"))
        else:  
            self.device = torch.device("cpu")
            self = self.to(torch.device("cpu"))
    
    def create_structure(self) -> None:
        """
        Define the layers, criterion and optimizer that will make up the model
        """
        hidden_layers = [128,64]
        output_size = 10        
        self.Linear1 = nn.Linear(self.input_size, hidden_layers[0])
        self.Linear2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.Linear3 = nn.Linear(hidden_layers[1], output_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.003)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define the forward pass of the neural network 
        """
        x = functional.relu(self.Linear1(x))
        x = functional.relu(self.Linear2(x))
        x = functional.log_softmax(self.Linear3(x), dim = 1)
        return x
    
    def load_data(self, batch_size: int = 64) -> None:
        """
        Load training data 
        """
        fashionMNIST = pd.read_csv('data/fashion-mnist_train.csv')
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainDataset = NeuralNetworkDataset(fashionMNIST,transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainDataset, batch_size=batch_size, shuffle=True)
        self.input_size = 28 * 28

    def test_accuracy(self, batch_size: int = 64) -> Tuple[str, float]:
        """
        Run through all images and make predictions then print the accuracy
        """
        testloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=True)
        accuracy = 0
        for image, labels in testloader:
            # Flatten the Image from 28*28 to 2352 column vector
            image = image.view(image.shape[0], -1)
            # Use GPU if available by assigning tensors to it
            image = image.to(self.device)
            labels = labels.to(self.device)   
            with torch.no_grad():
                probabilities = torch.exp(self(image.float()))
                # detach from gpu and convert to numpy arrays for processing
                probabilities = probabilities.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                for probabilityArr, label in zip(probabilities, labels):
                    # Get max of probability
                    MaxProbability = 0
                    MaxProbIndex = None 
                    i = 0
                    for probability in probabilityArr:
                        if probability > MaxProbability:
                            MaxProbability = probability
                            MaxProbIndex = i
                        i += 1            
                    # See if MaxProbability aligns with actual label 
                    if (MaxProbIndex == label):
                        accuracy += 1
        return ("Test Accuracy: " + str((accuracy / len(self.test_dataset)) * 100) + "%", (accuracy / len(self.test_dataset)))
    
    @staticmethod
    def view_classify(img, probabilities: torch.Tensor) -> None:
        """
        Take a single image with predicted class probabilities and display the image next to the class predictions
        """
        probabilities = probabilities.data.numpy().squeeze()
        len(probabilities)
        fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
        ax1.imshow(img.resize_(1, 28, 28).cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=255)
        ax1.axis('off')
        ax2.barh(np.arange(10), probabilities)
        ax2.set_aspect(0.1)
        ax2.set_yticks(np.arange(10))
        ax2.set_yticklabels(np.arange(10))
        ax2.set_title('Class Probability')
        ax2.set_xlim(0, 1.1)
        plt.tight_layout()
        plt.show()
    
    def save_model_checkpoint(self, filename = 'SavedModel'):
        """
        Save model for training and inference
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, filename)
        print("Model Saved")

    def load_model_checkpoint(self, filename = 'SavedModel'):
        """
        Load model for training and inference
        """
        checkpoint = torch.load(filename)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Model Loaded")
    
    @classmethod
    def print_model_state(cls) -> None: 
        """
        Print model state dict and optimizer state dict
        """
        # Print model's state_dict
        print("Model's state_dict:")
        for param_tensor in cls.model.state_dict():
            print(param_tensor, "\t", cls.model.state_dict()[param_tensor].size())
        # Print optimizer's state_dict
        print("Optimizer's state_dict:")
        for var_name in cls.optimizer.state_dict():
            print(var_name, "\t", cls.optimizer.state_dict()[var_name])
    
    def find_run_number(self) -> int:
        """
        Look through the folders in runs and determine the current run number
        """
        RunNum = -1
        for dirName, subdirList, fileList in os.walk("Runs"):
            if "Run_" in dirName:
                tempRunStr = dirName.replace("Runs\\Run_","")
                if (try_parse_int(tempRunStr)[0]):
                    tempRunNumber = try_parse_int(tempRunStr)[1]
                    if tempRunNumber > RunNum:
                        RunNum = tempRunNumber
        RunNum += 1
        self.RunNum = RunNum
        return self.RunNum

    def test_inference(self):
        # Getting the image to test
        images, labels = next(iter(self.trainloader)
        )# Flatten the image to pass in the model
        img = images[0].view(1, 784)
        img = img.to(self.device)
        # Turn off gradients to for greater performance, unneccassary for inference
        with torch.no_grad():
            probabilities = torch.exp(self(img.float()))
            self.view_classify(img, probabilities.cpu())
    
    def train(self, epochs = 5, check_accuracy = True, verbose = 2, save_model = True) -> None: 
        """
        Train the model against the training dataset
        """
        print("Training started")
        ListOfAccuracy = []
        ListOfRunningLoss = []
        for e in range(epochs):
            running_loss = 0
            accuracy = 0
            for images, labels in self.trainloader:
                # Flatten the Image from 28*28 to 2352 column vector
                images = images.view(images.shape[0], -1)
                # Use GPU if available
                images = images.to(self.device)
                labels = labels.to(self.device)   
                # setting gradient to zeros
                self.optimizer.zero_grad()        
                outputs = self(images.float())
                loss = self.criterion(outputs, labels)
                # backward propagation
                loss.backward()
                # update the gradient to new gradients
                self.optimizer.step()
                running_loss += loss.item()
            else:
                if check_accuracy:
                    accuracyReport, accuracyF = self.test_accuracy()
                    ListOfAccuracy.append(accuracyF)
                    if (verbose == 2):
                        print(accuracyReport)
                print("Training loss: ",(running_loss/len(self.trainloader)))
                # Append to 
                ListOfRunningLoss.append(running_loss/len(self.trainloader))

        # Create log of runs in run folder
        self.find_run_number()
        runs_directory = "Runs\Run_" + str(self.RunNum)
        os.mkdir(runs_directory)
        i = 0
        while i < len(ListOfRunningLoss):
            log_file = open(runs_directory + '\Log.txt', 'a')
            log_file.write('Epoch: ' + str(i) + ' \n')
            log_file.write('Log: ' + str(ListOfRunningLoss[i]) + ' \n')
            if check_accuracy:
                log_file.write('Accuracy: ' + str(ListOfAccuracy[i]) + ' \n')
            i += 1
        # Save Model
        if save_model:
            self.save_model_checkpoint(runs_directory + '\SavedModel')
