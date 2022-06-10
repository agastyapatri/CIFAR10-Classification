# **CIFAR-10 Classification**
## To be Appended.
_This project is meant to be a playground for PyTorch._
CIFAR-10 is a image dataset. 


NOTES:
 09/06/2022: batching is not done yet. 
Figuring out the dimensionality of the tensors, to 
facilitate matrix mulitplication 

why cant we just use one channel to predict? why all 3 ?
If RGB points to a certain label, then shouldnt just R 
point to the same label. 

both NLLloss and CrossEntropyLoss give the same value of 
loss for the label

Major error encountered: the weights were getting an 
inplace error? 


The total number of operations to be performed are 
$$N_{epochs} * N_{datapoints} = 1000 * 50000 = 5,000,000$$

 **training cycle 1: Epoch 19 : Training Loss = 1.6187867632266602**

