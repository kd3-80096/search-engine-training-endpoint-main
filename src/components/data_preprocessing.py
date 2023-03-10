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

    def transformations(self):
        try:
            """
            Transformation Method Provides TRANSFORM_IMG object. Its pytorch's transformation class to apply on images.
            :return: TRANSFORM_IMG
            """
            TRANSFORM_IMG = transforms.Compose(
                [transforms.Resize(self.config.IMAGE_SIZE),
                 transforms.CenterCrop(256),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])]
            )

            return TRANSFORM_IMG
        except Exception as e:
            raise e

    def create_loaders(self, TRANSFORM_IMG):
        """
        The create_loaders method takes Transformations and create dataloaders.
        :param TRANSFORM_IMG:
        :return: Dict of train, test, valid Loaders
        """
        try:
            print("Generating DataLoaders : ")
            result = {}
            for _ in tqdm(range(1)):
                train_data = ImageFolder(root=self.config.TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
                test_data = ImageFolder(root=self.config.TEST_DATA_PATH, transform=TRANSFORM_IMG)
                valid_data = ImageFolder(root=self.config.TEST_DATA_PATH, transform=TRANSFORM_IMG)

                train_data_loader = DataLoader(train_data, batch_size=self.config.BATCH_SIZE,
                                               shuffle=True, num_workers=1)
                test_data_loader = DataLoader(test_data, batch_size=self.config.BATCH_SIZE,
                                              shuffle=False, num_workers=1)
                valid_data_loader = DataLoader(valid_data, batch_size=self.config.BATCH_SIZE,
                                               shuffle=False, num_workers=1)

                result = {
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
    dp = DataPreprocessing()
    loaders = dp.run_step()
    for i in loaders["train_data_loader"][0]:
        break
