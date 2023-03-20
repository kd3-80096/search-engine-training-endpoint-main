from src.components.data_preprocessing import DataPreprocessing ## Importing DataPreprocessing class 
from src.entity.config_entity import TrainerConfig ## importing TrainerConfig class 
from torch import nn ## ## importing the neural network module which provides a wide range of building blocks for constructing 
# deep neural networks, such as layers and activation functions.
import torch ## importing the pytorch library
import numpy as np ## importing numpy
from src.components.model import NeuralNet  ## defines a class called NeuralNet which inherits from the nn.Module
from typing import Dict ## typing.Dict is a type hint that indicates the use of a dictionary object with string 
#keys and any value type. This is used to specify the expected type of a function argument or return value.
from tqdm import tqdm ## tqdm is a Python library that provides a progress bar that can be added to for loops to indicate the progress of the loop.


class Trainer: ##The constructor initializes several instance variables based on the input parameters:
#loaders is a dictionary object that contains data loaders for the training, testing, and validation data sets.
#device is a string that specifies the device on which the model will be trained (e.g., "cpu" or "cuda").
    def __init__(self, loaders: Dict, device: str, net): #net is an instance of the neural network model that will be trained.
        self.config = TrainerConfig() # creating the instance of TrainerConfig class 
        self.trainLoader = loaders["train_data_loader"][0] #data loaders for the training value is a list containing the data loader object at index 0 
        self.testLoader = loaders["test_data_loader"][0] # data loaders for testing value is a list containing the data loader object at index 0 
        self.validLoader = loaders["valid_data_loader"][0] # data loaders for vaidation value is a list containing the data loader object at index 0 
        self.device = device # the device on which the model will be trained.
        self.criterion = nn.CrossEntropyLoss() #loss function used during training (in this case, cross-entropy loss).
        self.model = net.to(self.device) ## Moves the neural network to the device specified by device. If device is "cuda", the neural network will
#be moved to the GPU, otherwise it will be moved to the CPU.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4) #optimizer used during training (in this case, Adam with a learning rate of 1e-4).
        self.evaluation = self.config.Evaluation #boolean that indicates whether the model should be evaluated on the
#validation set after each epoch of training. Its value is taken from the self.config.Evaluation attribute.

    def train_model(self): ## training the model in pytorch
        print("Start training...\n") ## printing the message start training
        for epoch in range(self.config.EPOCHS): ## a training loop that trains the model for the 2 number of epochs
            print(f'Epoch Number : {epoch}')  # print the no of epoch running
            running_loss = 0.0  ##running_loss initializing with 0
            running_correct = 0  ## running_correct initializing with 0 where the predicted label matches the true label.
            for data in tqdm(self.trainLoader): #for loop is created to iterate over the batches in the training data loader (self.trainLoader).
                data, target = data[0].to(self.device), data[1].to(self.device)  #extracts the input data and corresponding 
    #target labels from the batch, and moves them to the device (CPU or GPU) specified in self.device.
                self.optimizer.zero_grad() #sets the gradients of all model parameters to zero, which is necessary before computing the gradients for a new batch of data.
                outputs = self.model(data) # applies the neural network model self.model to the input data data, which produces predicted output values.
                loss = self.criterion(outputs, target) # computes the loss between the predicted outputs and the true target labels, using the loss function self.criterion (which is initialized as nn.CrossEntropyLoss()).
                running_loss += loss.item() #updates the running loss by adding the value of the loss for this batch of data (which is obtained by calling the .item() method of the PyTorch tensor).
                _, preds = torch.max(outputs.data, 1) # line extracts the predicted class labels for each input in 
#the batch, by taking the argmax of the output probabilities along the second dimension (i.e., across the classes).
                running_correct += (preds == target).sum().item() #  line updates the running number of correct 
#predictions by adding the number of input-target pairs where the predicted label matches the true label.

                loss.backward() # backpropogation the gradients of the loss with respect to all model parameters, using PyTorch's automatic differentiation.
                self.optimizer.step() #updates the model parameters by taking a step in the direction of the 
#negative gradients, using the optimization algorithm specified in self.optimizer (which is initialized as torch.optim.Adam with a learning rate of 1e-4).

            loss = running_loss / len(self.trainLoader.dataset) #The len(self.trainLoader.dataset) gives the total number of examples in the training dataset.
            accuracy = 100. * running_correct / len(self.trainLoader.dataset) # running_correct is the total number of
#correctly predicted samples by the model in the epoch. It is divided by the total number of samples in the 
# training dataset (len(self.trainLoader.dataset)) and multiplied by 100 to get the accuracy percentage.

            val_loss, val_accuracy = self.evaluate() # The validation loss and accuracy are obtained by calling the evaluate method of the Trainer class.

            print(f"Train Acc : {accuracy:.2f}, Train Loss : {loss:.4f}, " #The training and validation loss and accuracy are printed to the console using an f-string
                  f"Validation Acc : {val_accuracy:.2f}, Validation Loss : {val_loss:.4f}")

        print("Training complete!...\n")

    def evaluate(self, validate=False):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        self.model.eval() #set the model to evaluation mode.
        """model.eval() sets the model to evaluation mode, which disables dropout and batch normalization layers.
        During training, dropout and batch normalization layers are used to regularize the model and improve its
        generalization ability by preventing overfitting. However, during inference or evaluation, we want the model
        to produce deterministic outputs, so we turn off dropout and batch normalization. This is achieved by calling 
        model.eval() before making predictions.In addition to disabling dropout and batch normalization, calling 
        model.eval() also sets the requires_grad attribute of all model parameters to False, which disables gradient
        computation. This makes the model more memory-efficient, since we don't need to keep track of gradients during 
        evaluation."""
        val_accuracy = [] # initialize empty lists for storing the validation accuracy.
        val_loss = [] ## initialize empty lists for storing the val_loss.

        loader = self.testLoader if not validate else self.validLoader #select the test or validation data loader 
#based on the validate flag. If validate is False, the test loader is selected; otherwise, the validation loader is selected.

        with torch.no_grad(): # temporarily set all requires_grad flags to False to save memory during inference.
            for batch in tqdm(loader): # iterate over the batches in the selected data loader testLoader or validLoader
                img = batch[0].to(self.device) # move the input data to the selected device.
                labels = batch[1].to(self.device) #  move the target labels to the selected device.
                logits = self.model(img) #make predictions using the model.
                loss = self.criterion(logits, labels) #compute the loss between the predicted logits and the target labels.
                val_loss.append(loss.item()) # add the loss value to the val_loss list.
                preds = torch.argmax(logits, dim=1).flatten() #compute the predicted class labels from the logits and flatten them into a 1D tensor.

                accuracy = (preds == labels).cpu().numpy().mean() * 100 #compute the accuracy by comparing the 
#predicted class labels with the target labels,converting the result to a numpy array, taking the mean, and multiplying by 100 to get a percentage.
                val_accuracy.append(accuracy) #add the accuracy value to the val_accuracy list.

        val_loss = np.mean(val_loss) #compute the mean validation loss over all batches.
        val_accuracy = np.mean(val_accuracy) # compute the mean validation accuracy over all batches.

        return val_loss, val_accuracy  #return the computed validation loss and accuracy.

    def save_model_in_pth(self): # function saves the trained PyTorch model as a state dictionary in a .pth file format. 
        model_store_path = self.config.MODEL_STORE_PATH # path and file name of the model are defined in self.config.MODEL_STORE_PATH
        print(f"Saving Model at {model_store_path}") #state_dict() method returns a dictionary containing the model's weights and biases. 
        torch.save(self.model.state_dict(), model_store_path) # torch.save() function then saves this dictionary to
        #a file. The function also prints the path where the model is being saved.


if __name__ == "__main__": ## main code that runs the entire pipeline
    dp = DataPreprocessing() #creates an instance of the DataPreprocessing class, which is responsible for loading and preprocessing the data.
    loaders = dp.run_step() #calls the run_step method of the DataPreprocessing class, which returns a dictionary of data loaders.
    trainer = Trainer(loaders, "cpu", net=NeuralNet())  #creates an instance of the Trainer class, which is responsible for training and evaluating the model.
    trainer.train_model() # calls the train_model method of the Trainer class, which trains the model.
    trainer.evaluate(validate=True) #  calls the evaluate method of the Trainer class, which evaluates the model on the validation set.
    trainer.save_model_in_pth() # calls the save_model_in_pth method of the Trainer class, which saves the trained model to a file in the .pth format.



