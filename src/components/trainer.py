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
            running_correct = 0  ## running_correct initializing with 0
            for data in tqdm(self.trainLoader): #for loop is created to iterate over the batches in the training data loader (self.trainLoader).
                data, target = data[0].to(self.device), data[1].to(self.device) 
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                running_loss += loss.item()
                _, preds = torch.max(outputs.data, 1)
                running_correct += (preds == target).sum().item()

                loss.backward()
                self.optimizer.step()

            loss = running_loss / len(self.trainLoader.dataset)
            accuracy = 100. * running_correct / len(self.trainLoader.dataset)

            val_loss, val_accuracy = self.evaluate()

            print(f"Train Acc : {accuracy:.2f}, Train Loss : {loss:.4f}, "
                  f"Validation Acc : {val_accuracy:.2f}, Validation Loss : {val_loss:.4f}")

        print("Training complete!...\n")

    def evaluate(self, validate=False):
        """After the completion of each training epoch, measure the model's performance
        on our validation set.
        """
        self.model.eval()
        val_accuracy = []
        val_loss = []

        loader = self.testLoader if not validate else self.validLoader

        with torch.no_grad():
            for batch in tqdm(loader):
                img = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                logits = self.model(img)
                loss = self.criterion(logits, labels)
                val_loss.append(loss.item())
                preds = torch.argmax(logits, dim=1).flatten()

                accuracy = (preds == labels).cpu().numpy().mean() * 100
                val_accuracy.append(accuracy)

        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)

        return val_loss, val_accuracy

    def save_model_in_pth(self):
        model_store_path = self.config.MODEL_STORE_PATH
        print(f"Saving Model at {model_store_path}")
        torch.save(self.model.state_dict(), model_store_path)


if __name__ == "__main__":
    dp = DataPreprocessing()
    loaders = dp.run_step()
    trainer = Trainer(loaders, "cpu", net=NeuralNet())
    trainer.train_model()
    trainer.evaluate(validate=True)
    trainer.save_model_in_pth()
