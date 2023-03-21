from src.components.data_preprocessing import DataPreprocessing ##importing the DataPreprocessing class 
from src.entity.config_entity import ImageFolderConfig, EmbeddingsConfig #importing the ImageFolderConfig, EmbeddingsConfig class 
from src.utils.database_handler import MongoDBClient ## importing the MongoDBClient class 
from torch.utils.data import Dataset, DataLoader ## importing An abstract class representing a :class:`Dataset`.
## also importing Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
from src.components.model import NeuralNet ## importing the class NeuralNet
from typing import List, Dict ## list and dictionary importing from typing library
from torchvision import transforms #provides various image transformations such as resizing, cropping, normalization, and data augmentation.
from collections import namedtuple # is a factory function for creating tuple subclasses with named fields.
from PIL import Image # provides functions to open, manipulate, and save image files.
import torch #  is the main PyTorch package used for building and training deep learning models.
from torch import nn #provides various neural network layers and modules.
import pandas as pd # importing the pandas as pd
from tqdm import tqdm # to see the progess of the process we import the tqdm
import numpy as np ## importing numpy as np
import json ## The json module is a built-in Python module that provides methods for encoding and decoding JSON (JavaScript Object Notation) data. 
import os ## for the os operations
from pathlib import Path #Pathlib is a Python module that provides an object-oriented interface to work with file system paths

ImageRecord = namedtuple("ImageRecord", ["img", "label", "s3_link"]) ##code creates a new named tuple called ImageRecord
#with three fields: img, label, and s3_link. The first argument is the name of the new namedtuple and the second 
# argument is a list of field names for the new namedtuple. The fields are used to store information about each 
# image, including the image data itself (img), the corresponding label (label), and a link to the S3 storage 
# location for the image (s3_link).


class ImageFolder(Dataset):
    """This class is a PyTorch Dataset that reads image files from a directory and preprocesses them into tensors using
the transforms module from the torchvision library. The class reads the images, labels, and S3 links from the file
system, stores them in a list of ImageRecord named tuples, and provides the __len__ and __getitem__ methods to
     access the data."""
    def __init__(self, label_map: Dict): #Defines a constructor method for the ImageFolder class that takes in a dictionary label_map as input.
        self.config = ImageFolderConfig() # Initializes an instance of the ImageFolderConfig class and assigns it to the config attribute of the ImageFolder class. 
        self.config.LABEL_MAP = label_map #Assigns the input label_map dictionary to the LABEL_MAP attribute of the config object.
        self.transform = self.transformations() # Calls the transformations() method and assigns the returned transformation pipeline to the transform attribute of the ImageFolder class.
        self.image_records: List[ImageRecord] = [] # Initializes an empty list image_records with the data type of List[ImageRecord].
        self.record = ImageRecord # assigns the tupleImageRecord to the record attribute of the self.

        file_list = os.listdir(self.config.ROOT_DIR) # Retrieves a list of all the files and directories present in
        #the ROOT_DIR attribute of the config object

        for class_path in file_list: ## looping to the retrived files from file_list
            path = os.path.join(self.config.ROOT_DIR, f"{class_path}") # Joins the ROOT_DIR and the current directory 
            #path to get the absolute path of the current directory.
            images = os.listdir(path) # Retrieves a list of all the image filenames present in the current directory.
            for image in tqdm(images):#  Loops through each image filename in the current directory.
                image_path = Path(f"""{self.config.ROOT_DIR}/{class_path}/{image}""") #Constructs the absolute path of the current image.
                self.image_records.append(self.record(img=image_path, #Creates an ImageRecord object with the img, label, and s3_link attributes, 
                label=self.config.LABEL_MAP[class_path],#where img is the current image absolute path, label is the class label assigned to the current image directory
                s3_link=self.config.S3_LINK.format(self.config.BUCKET, class_path,#and s3_link is the URL of the image file in an S3 bucket.
                image))) #The ImageRecord object is then appended to the image_records list.


    def transformations(self): ## The transformations method defines the image transformations to be applied to each image.
        TRANSFORM_IMG = transforms.Compose( #the transforms.Compose function from PyTorch to apply a series of transformations in a pipeline.
            [transforms.Resize(self.config.IMAGE_SIZE), #resizing the image to a fixed size 256
             transforms.CenterCrop(self.config.IMAGE_SIZE), # center cropping it to the same size
             transforms.ToTensor(), # converting it to a PyTorch tensor
             transforms.Normalize(mean=[0.485, 0.456, 0.406], #normalizing it using the specified mean and standard deviation.
                                  std=[0.229, 0.224, 0.225])]
        )

        return TRANSFORM_IMG ## returning the transformed image

    def __len__(self):  #The __len__ method returns the length of the dataset, which is the number of image records in the dataset.
        return len(self.image_records)

    def __getitem__(self, idx): #The __getitem__ method is called when an item is retrieved from the dataset using indexing.
        record = self.image_records[idx] # 
        images, targets, links = record.img, record.label, record.s3_link #The image, target (label), and S3 link are retrieved from the record and the image is opened using PIL.Image.open. 
        images = Image.open(images)

        if len(images.getbands()) < 3: # If the image has fewer than three bands, it is converted to RGB. 
            images = images.convert('RGB') 
        images = np.array(self.transform(images)) #The image is then transformed using the transformations method, 
        targets = torch.from_numpy(np.array(targets)) #converted to a PyTorch tensor.
        images = torch.from_numpy(images)

        return images, targets, links #and returned along with the target and S3 link.


class EmbeddingGenerator:
    """ This class loads a pre-trained neural network model from a saved file, removes the last layer, and applies
    the model to the images in the ImageFolder dataset to generate embeddings. The embeddings are then saved to a 
    MongoDB database along with their labels and S3 links."""
    def __init__(self, model, device): #initialization method that takes two arguments, model and device.
        self.config = EmbeddingsConfig() # 
        self.mongo = MongoDBClient() # 
        self.model = model 
        self.device = device 
        self.embedding_model = self.load_model()
        self.embedding_model.eval()

    def load_model(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.config.MODEL_STORE_PATH, map_location=self.device))
        return nn.Sequential(*list(model.children())[:-1])

    def run_step(self, batch_size, image, label, s3_link):
        records = dict()

        images = self.embedding_model(image.to(self.device))
        images = images.detach().cpu().numpy()

        records['images'] = images.tolist()
        records['label'] = label.tolist()
        records['s3_link'] = s3_link

        df = pd.DataFrame(records)
        records = list(json.loads(df.T.to_json()).values())
        self.mongo.insert_bulk_record(records)

        return {"Response": f"Completed Embeddings Generation for {batch_size}."}


if __name__ == "__main__":
    dp = DataPreprocessing()
    loaders = dp.run_step()

    data = ImageFolder(label_map=loaders["valid_data_loader"][1].class_to_idx)
    dataloader = DataLoader(dataset=data, batch_size=64, shuffle=True)
    embeds = EmbeddingGenerator(model=NeuralNet(), device="cpu")

    for batch, values in tqdm(enumerate(dataloader)):
        img, target, link = values
        print(embeds.run_step(batch, img, target, link))
