"""
Classification on the CIFAR-10 dataset with PyTorch
"""
import torch, torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils import tensorboard
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

    def __init__(self, input_size, output_size, learning_rate, momentum, num_epochs, batches, weightdecay):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.epochs = num_epochs
        self.criterion_1 = nn.CrossEntropyLoss()
        self.criterion_2 = nn.NLLLoss()
        self.batches = batches
        self.weightdecay = weightdecay

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
            plt.subplot(1, idx, i+1)
            plt.imshow(images[i][1,:,:])
            plt.show()

    """-----------------------------------------------------------------------------------------------------------------
    3. Defining the Network  
    -----------------------------------------------------------------------------------------------------------------"""

    def network(self, loader):
        """
        Defining the NN that will be trained
        """
        dataiter = iter(loader)
        tensor, labels = dataiter.next()

        # input is now shaped [1, 3, 1024]
        input = torch.flatten(tensor, 1)

        hidden_sizes = [1024, 128, 64]

        model = nn.Sequential(
            # Input Layer
            nn.Linear(self.input_size, hidden_sizes[0]),
            nn.ReLU(),
            # Hidden Layer 1
            nn.Linear(hidden_sizes[0], hidden_sizes[2]),
            nn.ReLU(),

            # Hidden Lauer 3
            nn.Linear(hidden_sizes[2], self.output_size),
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
        Function to perform the training step. Also calculates the training accuracy for each epoch
        :param loader: the train loader
        """
        images, labels = next(iter(train))
        model = self.network(loader)[1]
        training_loss = []
        training_accuracy_array = []
        # Defining the optimizer for the model
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate,
                                    momentum=self.momentum, weight_decay=self.weightdecay)

        # Looping over the number of epochs
        for epoch in range(self.epochs):
            running_loss = 0.0
            correct_count = 0
            incorrect_count = 0

            # Looping over all the images
            for i, datapoint in enumerate(loader):
                image , true_label = datapoint
                inputs = torch.flatten(image, 1)

                optimizer.zero_grad()
                prediction = model(inputs)
                predicted_labels = torch.max(prediction, 1)[1]

                # Actual training pass
                with torch.autograd.set_detect_anomaly(True):
                    # finding loss
                    loss = self.criterion_1(prediction, true_label)
                    # backpropagation
                    loss.backward(retain_graph = True)
                    # updating weights
                    optimizer.step()

                running_loss += loss.item()

                # Calculating the correctly and incorrectly predicted labels for training.
                bool_array = np.array(predicted_labels == true_label)
                num_correct, num_incorrect = np.count_nonzero(bool_array), self.batches - np.count_nonzero(bool_array)

                training_loss.append(running_loss/len(train))
                correct_count += num_correct
                incorrect_count += num_incorrect




            else:
                training_accuracy = correct_count / (correct_count + incorrect_count)
                training_accuracy_array.append(training_accuracy)

                print(f"Epoch {epoch} : Training Loss = {running_loss/len(train)}. Training Accuracy = {round(training_accuracy*100, 6)} %")

        print("Training Done!")

        # Creating numpy arrays for the two metrics.
        training_loss = np.array(training_loss)
        training_accuracy_array = np.array(training_accuracy_array)

        return training_loss, training_accuracy_array




    """-----------------------------------------------------------------------------------------------------------------
    5. Testing the Model    
    -----------------------------------------------------------------------------------------------------------------"""

    def test_model(self, loader):
        """
        Function to enable testing of data
        :param loader: testloader
        """

        images, labels = next(iter(loader))

        # Calculating the number of correct and incorrect labels
        correct_count = 0
        incorrect_count = 0


        with torch.no_grad():
            # Looping over the batches
            for i, data in enumerate(loader):
                image, true_labels = data
                flattened_image = torch.flatten(image, 1)

                model = self.network(loader)[1]
                prediction = model(flattened_image)
                predicted_labels = torch.max(prediction, 1)[1]

                # Bool array is an array of Truth values, and the code below counts how many are True (correctly predicted)
                bool_array = np.array(predicted_labels == true_labels)
                num_correct, num_incorrect = np.count_nonzero(bool_array), self.batches - np.count_nonzero(bool_array)

                correct_count += num_correct
                incorrect_count += num_incorrect

            accuracy = correct_count / (correct_count + incorrect_count)
            print(f"Testing Accuracy of the Model is: {round(accuracy * 100, 6)} % ")
            return accuracy




    """-----------------------------------------------------------------------------------------------------------------
    6. Plotting the Results
    -----------------------------------------------------------------------------------------------------------------"""
    def learning_curves(self, training_loss):
        """
        Function to plot the loss curves
        :param training_loss: array
        :return:
        """
        pass



#----------------------------------------------------------------------------------------------------------------------#

if __name__ == "__main__":

    """-----------------------------------------------------------------------------------------------------------------
    0. Preliminary Operations and Functions 
    -------------------------------------------------------------------Commit 2: ".gitignore" added to /data/ to ignore the large data files.----------------------------------------------"""
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
        loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size = 16)
        return loader


    """-----------------------------------------------------------------------------------------------------------------
    1. Running the Code
        ~ train is the DataLoader object for the training set
        ~ validation is the DataLoader object for the validation set.
    -----------------------------------------------------------------------------------------------------------------"""
    train = load_data(option="train")
    validation = load_data(option="validation")
    len_train = len(list(enumerate(train)))
    len_val = len(list(enumerate(validation)))


    model1 = Model(input_size=3072, output_size=10, learning_rate=1e-3, momentum=0.9, num_epochs=30, batches=16, weightdecay=1e-5)


    def RunModel():
        """
        Runner Code to perform each step defined in the class Model. This function also Plots the training/testing loss + accuracy
        train_loss.shape = num_epochs * num_batches
        """
        train_loss, train_accuracy = model1.train_model(train)
        validated = model1.test_model(validation)


        # steps_X = np.array(range(model1.ep\ochs))
        # train_accuracy_Y1 = train_accuracy
        # train_loss_Y2 = train_loss
        #
        # # Plotting the Learning Curves of the Model
        # fig, axs = plt.subplot_mosaic([ ["A", "C"],["B", "C"]], constrained_layout=True)






