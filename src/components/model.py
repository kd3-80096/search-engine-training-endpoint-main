from src.entity.config_entity import ModelConfig ## importing the class named ModelConfig
from torch import nn ## importing the neural network module which provides a wide range of building blocks for constructing 
# deep neural networks, such as layers and activation functions.
import torch ## importing torch The torch package contains data structures for multi-dimensional tensors


class NeuralNet(nn.Module): ## defines a class called NeuralNet which inherits from the nn.Module
    def __init__(self): #The __init__ method defines the architecture of the neural network using the nn.Sequential container, which allows the layers to be defined in a sequential order.
        super().__init__() #The super().__init__() line calls the constructor of the nn.Module class to properly initialize the NeuralNet object.
        self.config = ModelConfig() ## creating an instance of class ModelConfig
        self.base_model = self.get_model() ## creating instance of get_model method so as to call the model parameters
        """"Defines three convolutional layers using the nn.Conv2d() class from PyTorch. These layers have 512, 32, and 16
        input channels respectively, and output 32, 16, and 4 channels respectively. The kernel size for each layer is 
        (3,3), the stride is (1,1), and the padding is (1,1)."""
        self.conv1 = nn.Conv2d(512, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.flatten = nn.Flatten() ## # a flattening layer, which flattens the output of the convolutional layers into a 1D vector that can be processed by a fully connected layer.
        self.final = nn.Linear(4 * 8 * 8, self.config.LABEL) ## Defines a fully connected layer using the nn.Linear() 
#class from PyTorch. The input to this layer is a tensor with dimensions (4, 8, 8) which corresponds to the output 
# dimensions of the final convolutional layer. The output dimension of this layer is equal to the 101 LABEL attribute 
# of the self.config object.

    def get_model(self):
        torch.hub.set_dir(self.config.STORE_PATH) ##  sets the directory where the pre-trained models downloaded via torch.hub.load() will be stored.
        model = torch.hub.load( ## loading the model configurations
            self.config.REPOSITORY, # REPOSITORY variable to the PyTorch vision library version v0.10.0.
            self.config.BASEMODEL,  ## algorithm used is resnet-18
            pretrained=self.config.PRETRAINED # pretraining is set to true
        )
        return nn.Sequential(*list(model.children())[:-2]) """expression returns all but the last two layers of the pre-trained model. The last two layers of the model
        are typically a fully connected layer and a softmax activation layer, which are used for classification tasks.
        By removing these layers, the modified model can be used as a feature extractor that maps an input image to a 
        fixed-length feature vector. This feature vector can be used for task image similarity search. The modified model is returned by the get_model() method."""

    def forward(self, x):
        """The forward method defines the forward pass of the neural network, which takes an input tensor x and applies the
        layers defined in self.model."""
        x = self.base_model(x) #Applies the layers defined in self.base_model (which is a pre-trained model loaded via torch.hub.load()) to the input tensor x. The output tensor is assigned to x.
        x = self.conv1(x) #Applies the first convolutional layer to the output tensor from the previous step. The output tensor is assigned to x.
        x = self.conv2(x) #Applies the second convolutional layer to the output tensor from the previous step. The output tensor is assigned to x.
        x = self.conv3(x) # Applies the third convolutional layer to the output tensor from the previous step. The output tensor is assigned to x.
        x = self.flatten(x) #Flattens the output tensor from the previous step into a 1D tensor. The flattened tensor is assigned to x.
        x = self.final(x) #Applies the final linear layer to the flattened tensor from the previous step. The output tensor is assigned to x.
        return x # Returns the output tensor from the final layer, which represents the neural network's prediction for the input tensor x.


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu" # Determines whether the current system has a GPU that 
#can be used to accelerate the computation. If a GPU is available, the device is set to "cuda", otherwise it is set to "cpu".
    net = NeuralNet() # Creates an instance of the NeuralNet class, which defines the architecture of the neural network.
    net.to(device) # Moves the neural network to the device specified by device. If device is "cuda", the neural network will
#be moved to the GPU, otherwise it will be moved to the CPU.In summary, this code initializes an instance of the NeuralNet 
# class and sets the device (CPU or GPU) that will be used to execute the neural network.






