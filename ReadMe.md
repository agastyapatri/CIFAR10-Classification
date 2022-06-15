# **CIFAR-10 Classification**

_The CIFAR-10 dataset consists of 60,000 32 x 32 colour images in 10 classes,
with 6,000 images per class. There are 50,000 training images and 10,000 test images._



## Description of the Model used, Finally.
The Neural Network consists of the following:
1. **Convolution Layer:** Input channels -> 3, Output channels -> 6. Pooling Applied, with ReLU activation
2. **Convolution Layer** Input Channels -> 6, Output Channels -> 16. Pooling Applied, with ReLU activation
3. **Linear Layer:** Input Channels -> 400, Output Channels -> 120. ReLU activation 
4. **Linear Layer** Input Channels -> 120, Output Channels -> 84. ReLU activation 
5. **Linear Layer** Input Channels -> 84, Output Channels -> 10. 

## Configuration 
* Number of Epochs = 50
* Batch Size = 16
* Loss Function = Cross Entropy Loss
* Optimizer = Adam 
* Learning Rate = 0.0005 
* Momentum = 0.9

## Final Results
* Training Accuracy = 80.56 % 
* Training Loss = 0.523



## 1. Training Cycle 1

TC 1 was a MLP with 2 hidden layers and ReLU activation

TC 1 had no batching. The training took ~ 8 hrs. 
Information regarding the validation of the model is not 
available. 

 **training cycle 1: Epoch 19 : Training Loss = 1.6187867632266602**


## 2. Training Cycle 2
### 2.1 Immediate Steps Taken to Reduce Training Loss after TC1. 
1. **Batching**: To increase the training efficiency, a batch size of 64 is selected
   After batching, the train loader is of shape (782, 2) with each row representing a batch of 64, and the columns
   the image and the label respectively.
   * Before Flattening, the shape of each instance of the dataloader is (64 x 3 x 32 x 32). After flattening, the shape is
     (64, 3072). 

4. **Epochs**: The number of epochs were increased to 30. Chosen arbitrarily. 

### 2.2 Observations on the Training Cycle 2.
* Training happened orders of magnitude faster than TC 1.
* With the changes made above, training loss settled at **1.1677925**, with an accuracy of  **60 %**
* The testing accuracy, however, is merely **9.892516 %** ! Does this mean that the model has overfit?!
* With reasonable condfidence it can be said that the model has **Overfit**

### 2.3 Changes and Experiments to improve Training accuracy and Overfitting.
_**Hyperparameter Optimization**_

1. **Reducing the number of epochs** from 30 to 20
   * Done to see if the model is overfitting.
   
   * _Observations_: The Training accuracy reduced to **53.51 %**, 
    and the testing accuracy reduced to **9.703 %**


2. **Increasing the number of epochs** from 30 to 40.
   * Done as due-diligence. A reduction in the testing accuracy is expected
   * _Observations:_ Training Accuracy settled at **62.72 %**
    and Testing Accuracy increased to **9.98 %**

    
3. **Reducing the Batch Size** from 64 to 16.
   Larger Batch Sizes train quicker, but are less accurate. 
   * _Observations:_ The training Accuracy increased to **66.584 %**
    and the Testing Accuracy increased to a dismal **10.26 %**
   * Although this is an improvement, it is computationally expensive.
    Further experimentation will use a batch size of 64 to hasten the training process, 
     and the final batch size will be **16.**


4. **Changing the Learning Rate**
   * Increasing from **0.001** to **0.01** gives a Training Accuracy of **64.402 %**
    and Increasing the Testing accuracy to **10.061 %**


5. **Increasing the Learning Rate and Decreasing the Batch Size:** 
   Only changes in the Learning Rate and the Batch size has resulted in an increase 
   in training accuracy. 
   * The model actually performs worse! Training accuracy = **45.882 %** 

   * With batch size = 32, learning rate = 0.01 gives training accuracy = **59.69 %**
     and Testing Accuracy of **9.93 %**


6. **Changing the Activation Function of the Hidden Layers**
   * Changing all from ReLU to Sigmoid : The Worst starting training accuracy of all things tried.

    
7. **Increasing the number of Hidden Layers**
    * One extra Linear hidden layer with 128 nodes is added. It still uses a RelU activation.
      
      _Observation:_ Training Accuracy of *58.374 %* and Testing Accuracy of *9.196 %*


9. **Dropout:**
   Adding dropout layers with p = 0.25 gives worse performance. 

10. **Changing the Model itself.**
    * Using a CNN to re-cast the model. This was the 
     last resort when it came to optimization



