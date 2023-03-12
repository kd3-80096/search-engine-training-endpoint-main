from src.entity.config_entity import DataPreprocessingConfig ## importing the DataPreprocessingConfig class 
from torchvision.datasets import ImageFolder  ##from torchvision.datasets import ImageFolder is a Python import 
#statement that imports the ImageFolder class from the torchvision.datasets module.The ImageFolder class is a
#  dataset class that can be used to load a set of images from a directory structure where each subdirectory 
# represents a different class. This is a common format for image classification datasets, where each image 
# is assigned a label that corresponds to its class.
from torch.utils.data import DataLoader# : a PyTorch utility for loading and batching data from a dataset,which
#makes it easier to iterate over the  data during training.
from torchvision import transforms## The transforms module provides a set of common image transformations that 
#can be used for data augmentation and preprocessing in computer vision tasks. These transformations can be applied
# to input images to make the training process more robust and improve the performance of machine learning models.
from tqdm import tqdm ## to show the progressbar


class DataPreprocessing: ## contains the methods that have image preprocessing steps in them
    def __init__(self): # __init__(self) is the constructor method of the class, which is called when an instance of the class is created.
        self.config = DataPreprocessingConfig() # DataPreprocessingConfig class is stored in self.config instance

    def transformations(self): ## This method  the transforms module provides a set of common image transformations 
        try:
            """
            Transformation Method Provides TRANSFORM_IMG object. Its pytorch's transformation class to apply on images.
            :return: TRANSFORM_IMG
            """
            TRANSFORM_IMG = transforms.Compose( ## Composes several transforms together.
                [transforms.Resize(self.config.IMAGE_SIZE),## resize the image to shape to 256
                 transforms.CenterCrop(256), ## This transformation takes a PIL (Python Imaging Library) image as input and 
#returns a new image of size (256, 256) by cropping the input image from its center. The transformation ensures that the center of the image is preserved while removing the outer edges of the image.
                 transforms.ToTensor(), ##Convert a PIL Image or numpy.ndarray to tensor. This transform does not support torchscript.
#Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], #data preprocessing technique used in deep learning models for computer vision tasks.
#This transformation is typically applied to the image data after it has been resized or cropped, and before it is fed into the neural network.
#The mean and std arguments are lists of length 3, corresponding to the mean and standard deviation of the pixel values for the red, green, and blue 
# channels, respectively. These values are usually calculated based on the training dataset and are used to normalize the pixel values of the input images so that they have a similar scale and range.
                                      std=[0.229, 0.224, 0.225])] #This operation scales the pixel values to a range of approximately -1 to 1, which is generally suitable for training deep neural networks.
            )

            return TRANSFORM_IMG ## return the transformed image parameters to be used in the create_loaders method input
        except Exception as e:
            raise e

    def create_loaders(self, TRANSFORM_IMG): ## this method will take the transformed images from the transformations method as input.
        """
        The create_loaders method takes Transformations and create dataloaders.
        :param TRANSFORM_IMG:
        :return: Dict of train, test, valid Loaders
        """
        try:
            print("Generating DataLoaders : ")
            result = {} #The function starts by creating an empty dictionary named result to store the resulting dataloaders.
            for _ in tqdm(range(1)): # In this case, the loop for _ in tqdm(range(1)): is simply a way to create a progress bar that will 
#iterate for one iteration only. The use of the range function with an argument of 1 is just a convenient way to create a loop that will run only once.
#The underscore _ is used as a variable name to indicate that we don't actually need to use the loop variable, and it is just there to ensure that the loop runs once.
                train_data = ImageFolder(root=self.config.TRAIN_DATA_PATH, transform=TRANSFORM_IMG) # uses the ImageFolder class from PyTorch to 
#create three datasets: train_data, test_data, and valid_data. Each dataset is initialized with a root directory 
# where the images are stored and the specified transformation TRANSFORM_IMG.
                test_data = ImageFolder(root=self.config.TEST_DATA_PATH, transform=TRANSFORM_IMG)
                valid_data = ImageFolder(root=self.config.TEST_DATA_PATH, transform=TRANSFORM_IMG)

                train_data_loader = DataLoader(train_data, batch_size=self.config.BATCH_SIZE,
                                               shuffle=True, num_workers=1)#method uses the DataLoader class to create dataloaders for each dataset.
                test_data_loader = DataLoader(test_data, batch_size=self.config.BATCH_SIZE,
                                              shuffle=False, num_workers=1) #Each dataloader is specified with a batch size (self.config.BATCH_SIZE), 
                valid_data_loader = DataLoader(valid_data, batch_size=self.config.BATCH_SIZE, #shuffle parameter to shuffle the data
                                               shuffle=False, num_workers=1) # and the number of worker processes used for loading the data (num_workers=1).

                result = {
                    """Finally, the method returns a dictionary result containing the three dataloaders for the 
                    training, testing, and validation datasets, along with their corresponding datasets."""

                    "train_data_loader": (train_data_loader, train_data),
                    "test_data_loader": (test_data_loader, test_data),
                    "valid_data_loader": (valid_data_loader, valid_data)
                }
            return result
        except Exception as e:
            raise e

    def run_step(self):
        try:
            """
            This methods calls all the private methods.
            :return: Response of Process
            """
            TRANSFORM_IMG = self.transformations()
            result = self.create_loaders(TRANSFORM_IMG)
            return result
        except Exception as e:
            raise e


if __name__ == "__main__":
    # Data Ingestion Can be replaced like this
    # https://aws.amazon.com/blogs/machine-learning/announcing-the-amazon-s3-plugin-for-pytorch/
    dp = DataPreprocessing() ## creating the instance of the class DataPreprocessing
    loaders = dp.run_step()
    for i in loaders["train_data_loader"][0]:#for loop iterates over the batches in the training dataloader by 
    #calling loaders["train_data_loader"][0] and breaking after the first batch using the break statement.
        break





