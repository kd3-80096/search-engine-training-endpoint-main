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

ImageRecord = namedtuple("ImageRecord", ["img", "label", "s3_link"])


class ImageFolder(Dataset):
    """This class is a PyTorch Dataset that reads image files from a directory and preprocesses them into tensors using
the transforms module from the torchvision library. The class reads the images, labels, and S3 links from the file
system, stores them in a list of ImageRecord named tuples, and provides the __len__ and __getitem__ methods to
     access the data."""
    def __init__(self, label_map: Dict): #Defines a constructor method for the ImageFolder class that takes in a dictionary label_map as input.
        self.config = ImageFolderConfig() # Initializes an instance of the ImageFolderConfig class and assigns it to the config attribute of the ImageFolder class. 
        self.config.LABEL_MAP = label_map
        self.transform = self.transformations()
        self.image_records: List[ImageRecord] = []
        self.record = ImageRecord

        file_list = os.listdir(self.config.ROOT_DIR)

        for class_path in file_list:
            path = os.path.join(self.config.ROOT_DIR, f"{class_path}")
            images = os.listdir(path)
            for image in tqdm(images):
                image_path = Path(f"""{self.config.ROOT_DIR}/{class_path}/{image}""")
                self.image_records.append(self.record(img=image_path,
                                                      label=self.config.LABEL_MAP[class_path],
                                                      s3_link=self.config.S3_LINK.format(self.config.BUCKET, class_path,
                                                                                         image)))

    def transformations(self):
        TRANSFORM_IMG = transforms.Compose(
            [transforms.Resize(self.config.IMAGE_SIZE),
             transforms.CenterCrop(self.config.IMAGE_SIZE),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]
        )

        return TRANSFORM_IMG

    def __len__(self):
        return len(self.image_records)

    def __getitem__(self, idx):
        record = self.image_records[idx]
        images, targets, links = record.img, record.label, record.s3_link
        images = Image.open(images)

        if len(images.getbands()) < 3:
            images = images.convert('RGB')
        images = np.array(self.transform(images))
        targets = torch.from_numpy(np.array(targets))
        images = torch.from_numpy(images)

        return images, targets, links


class EmbeddingGenerator:
    """ This class loads a pre-trained neural network model from a saved file, removes the last layer, and applies
    the model to the images in the ImageFolder dataset to generate embeddings. The embeddings are then saved to a 
    MongoDB database along with their labels and S3 links."""
    def __init__(self, model, device):
        self.config = EmbeddingsConfig()
        self.mongo = MongoDBClient()
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
