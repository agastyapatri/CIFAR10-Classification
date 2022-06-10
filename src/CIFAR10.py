"""
Classification on the CIFAR-10 dataset with PyTorch
"""
import torch, torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn.functional as functional
from torchvision import transforms
import os

"""---------------------------------------------------------------------------------------------------------------------
3. Visualizing the datapoints
5. Building the Neural Network
6. Training the NN + Backprop
7. Hyperparameter Optimization
---------------------------------------------------------------------------------------------------------------------"""

class Model(nn.Module):
    """
    The model which will be trained and tested
    """

    def __init__(self, input_size, output_size, learning_rate, momentum, epochs, batches):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = epochs
        self.criterion_1 = nn.CrossEntropyLoss()
        self.criterion_2 = nn.NLLLoss()
        self.batches = batches

    """-----------------------------------------------------------------------------------------------------------------
    2. Visualizing the Data 
    -----------------------------------------------------------------------------------------------------------------"""

    def visualize_data(self, loader):
        """
        Function to visualtize the image data. "dataset" is either the train or the test tuple in the runner main code.
        """
        idx = int(np.sqrt(self.batches))
        dataiter = iter(loader)
        images, labels = dataiter.next()

        for i in range(idx):
            plt.subplot(idx, idx, i+1)
            plt.imshow(images[i][1,:,:])
            plt.show()

    """-----------------------------------------------------------------------------------------------------------------
    3. Defining the Network  
    -----------------------------------------------------------------------------------------------------------------"""

    def network(self, loader):
        """
        Defining the CNN that will be trained
        """
        dataiter = iter(loader)
        tensor, labels = dataiter.next()

        # input is now shaped [1, 3, 1024]
        input = torch.flatten(tensor, 1)

        hidden_sizes = [1024, 64]

        model = nn.Sequential(
            # Input Layer
            nn.Linear(self.input_size, hidden_sizes[0]),
            nn.ReLU(),
            # Hidden Layer
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            # Output Lauer
            nn.Linear(hidden_sizes[1], self.output_size),
            nn.LogSoftmax(dim=1)
        )
        # The shape of the predictions are [1 x 10]
        predictions = model(input)
        return predictions, model

    """-----------------------------------------------------------------------------------------------------------------
    4. Training the Model + Backpropagation 
    -----------------------------------------------------------------------------------------------------------------"""

    def train_model(self, loader):
        """
        Function to perform the training step.
        :param loader: the train loader
        """
        images, labels = next(iter(train))
        model = self.network(train)[1]
        training_loss = []

        # Defining the optimizer for the model
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        # Looping over the number of epochs
        for epoch in range(self.epochs):
            running_loss = 0.0

            # Looping over all the images
            for i, datapoint in enumerate(loader):
                image , label = datapoint
                inputs = torch.flatten(image, 1)

                optimizer.zero_grad()
                prediction = model(inputs)

                # Actual training pass
                with torch.autograd.set_detect_anomaly(True):
                    # finding loss
                    loss = self.criterion_1(prediction, label)

                    # backpropagation
                    loss.backward(retain_graph = True)

                    # updating weights
                    optimizer.step()

                    running_loss += loss.item()
                    training_loss.append(running_loss/len(train))

            else:

                print(f"Epoch {epoch} : Training Loss = {running_loss/len(train)}")
        print("Training Done!")
        training_loss = np.array(training_loss)
        return training_loss


    """-----------------------------------------------------------------------------------------------------------------
    5. Testing the Model    
    -----------------------------------------------------------------------------------------------------------------"""

    def test_model(self, loader):
        """
        Function to enable testing of data
        :param loader: testloader
        """

        images, labels = next(iter(loader))
        predictions, model = self.network(loader)

        correct_count = 0
        incorrect_count = 0

        # Calculating the number of correct and incorrect labels
        with torch.no_grad():
            for i, data in enumerate(loader):
                image, label = data
                inputs = torch.flatten(image, 1)
                prediction = model(inputs)
                predicted_label = torch.max(prediction)

                if predicted_label == label:
                    correct_count += 1
                else:
                    incorrect_count += 1

        accuracy = (correct_count)/ (correct_count + incorrect_count)

        print(f"Accuracy of the Model = {accuracy*100}%")
        return accuracy, correct_count, incorrect_count

    """-----------------------------------------------------------------------------------------------------------------
    6. Plotting the Results
    -----------------------------------------------------------------------------------------------------------------"""
    def results(self, training_loss):
        """
        Function to plot the loss curves
        :param training_loss: array
        :return:
        """
        steps_X = range(self.epochs)
        targets_Y = training_loss

        plt.title("Training Loss of the Model")
        plt.plot(steps_X, targets_Y, "g-")
        plt.grid()
        plt.xlabel("Epochs / Training Steps")
        plt.ylabel(f"Training Loss ({self.criterion_1})")
        plt.show()



if __name__ == "__main__":

    """-----------------------------------------------------------------------------------------------------------------
    0. Preliminary Operations and Functions 
    -----------------------------------------------------------------------------------------------------------------"""
    PATH = "/home/agastya123/PycharmProjects/DeepLearning/CIFAR-10/"

    def load_data(option):
        """
        Function to load image data and return tensors. Resturns a DataLoader Object.
        """
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(0, 1)])

        if option == "train":
            dataset = torchvision.datasets.CIFAR10(os.path.join(PATH, "data/"), download=True, train=True,
                                                   transform=transform)
        elif (option == "validation" or option == "test"):
            dataset = torchvision.datasets.CIFAR10(os.path.join(PATH, "data/"), download=True, train=False,
                                                   transform=transform)
        else:
            print("option can only be train, validation and test!")

        # Data has not been batched yet
        loader = torch.utils.data.DataLoader(dataset, shuffle=True)
        return loader


    """-----------------------------------------------------------------------------------------------------------------
    1. Running the Code
        ~ train is the DataLoader object for the training set
        ~ validation is the DataLoader object for the validation set.
    -----------------------------------------------------------------------------------------------------------------"""
    train = load_data(option="train")
    validation = load_data(option="validation")

    model1 = Model(input_size=3072, output_size=10, learning_rate = 0.001, momentum = 0.9, epochs = 20, batches = 1)


    def RunModel():
        """
        Runner Code to perform each step defined in the Model Class This is a test
        """
        trained = model1.train_model(train)
        validated = model1.test_model(validation)
        result = model1.results(trained)

