from from_root import from_root ## importing the library from_root to get the root directory of the project
from dotenv import load_dotenv ## importing the load_dotenv that loads all variables in .env file
import os ## importing os for system operations


class DatabaseConfig: ## class name is DatabaseConfig

    def __init__(self): #__init__(self) is the constructor method of the class, which is called when an instance of the class is created.
        # Load environment variables from .env file
        load_dotenv() # method that loads  environment variables from a .env file.
        self.USERNAME: str = os.environ["DATABASE_USERNAME"] #os.environ["DATABASE_USERNAME"] assigns the value of 
        #the environment variable DATABASE_USERNAME to the USERNAME attribute of the instance.
        self.PASSWORD: str = os.environ["DATABASE_PASSWORD"]
        self.URL: str = os.getenv("MONGODB_URL_KEY") #os.getenv("MONGODB_URL_KEY") assigns the value of the 
        # environment variable MONGODB_URL_KEY to the URL attribute of the instance.
        self.DBNAME: str = "ReverseImageSearchEngine" # assigns the string "ReverseImageSearchEngine" to the DBNAME attribute of the instance.
        self.COLLECTION: str = "Embeddings" #  assigns the string "Embeddings" to the COLLECTION attribute of the instance.

    def get_database_config(self): # method of the class that returns a dictionary of the instance's attributes.
        return self.__dict__


class DataIngestionConfig: ## class name is DataIngestionConfig
    def __init__(self): #__init__(self) is the constructor method of the class, which is called when an instance of the class is created.
        self.PREFIX: str = "images/"  #assigns the string "images/" to the PREFIX attribute of the instance.
        self.RAW: str = "data/raw" # assigns the string "data/raw" to the RAW attribute of the instance.
        self.SPLIT: str = "data/splitted" # assigns the string "data/splitted" to the SPLIT attribute of the instance.
        self.BUCKET: str = "search-image619" # assigns the string "search-image619" to the BUCKET attribute of the instance
        self.SEED: int = 1337  
        """The term "seed" in this context typically refers to a random number generator seed. A random number generator
      uses a seed value to generate a sequence of seemingly random numbers. If two generators are seeded with the same 
      value, they will produce the same sequence of numbers.The purpose of setting a specific seed value is to ensure 
      that the random number generator produces the same sequence of numbers each time the program is run. This can
    be useful in situations where you want to replicate the results of a random process, such as when training a machine
      learning model. By setting the seed to a specific value, you can ensure that the random numbers used during training are the same each time, which can help you reproduce your results"""
        self.RATIO: tuple = (0.8, 0.1, 0.1)  # the ration in which the train,test and validation folders will be split

    def get_data_ingestion_config(self):
        return self.__dict__ #  method of the class that returns a dictionary of the instance's attributes.


class DataPreprocessingConfig:
    def __init__(self):
        self.BATCH_SIZE = 32
        self.IMAGE_SIZE = 256
        self.TRAIN_DATA_PATH = os.path.join(from_root(), "data", "splitted", "train")
        self.TEST_DATA_PATH = os.path.join(from_root(), "data", "splitted", "test")
        self.VALID_DATA_PATH = os.path.join(from_root(), "data", "splitted", "valid")

    def get_data_preprocessing_config(self):
        return self.__dict__


class ModelConfig:
    def __init__(self):
        self.LABEL = 101
        self.STORE_PATH = os.path.join(from_root(), "model", "benchmark")
        self.REPOSITORY = 'pytorch/vision:v0.10.0'
        self.BASEMODEL = 'resnet18'
        self.PRETRAINED = True

    def get_model_config(self):
        return self.__dict__


class TrainerConfig:
    def __init__(self):
        self.MODEL_STORE_PATH = os.path.join(from_root(), "model", "finetuned", "model.pth")
        self.EPOCHS = 2
        self.Evaluation = True

    def get_trainer_config(self):
        return self.__dict__


class ImageFolderConfig:
    def __init__(self):
        self.ROOT_DIR = os.path.join(from_root(), "data", "raw", "images")
        self.IMAGE_SIZE = 256
        self.LABEL_MAP = {}
        self.BUCKET: str = "search-image619"
        self.S3_LINK = "https://{0}.s3.ap-south-1.amazonaws.com/images/{1}/{2}"
        

    def get_image_folder_config(self):
        return self.__dict__


class EmbeddingsConfig:
    def __init__(self):
        self.MODEL_STORE_PATH = os.path.join(from_root(), "model", "finetuned", "model.pth")

    def get_embeddings_config(self):
        return self.__dict__


class AnnoyConfig:
    def __init__(self):
        self.EMBEDDING_STORE_PATH = os.path.join(from_root(), "data", "embeddings", "embeddings.ann")

    def get_annoy_config(self):
        return self.__dict__


class s3Config: ## name of class is s3Config

    def __init__(self): ## constructor method of the class, which is called when an instance of the class is created.
        self.ACCESS_KEY_ID = os.environ["ACCESS_KEY_ID"] # os.environ["ACCESS_KEY_ID"] assigns the value of the 
        #environment variable ACCESS_KEY_ID to the ACCESS_KEY_ID attribute of the instance.
        self.SECRET_KEY = os.environ["AWS_SECRET_KEY"] # 
        self.REGION_NAME = "ap-south-1" # assigns the string "ap-south-1" to the REGION_NAME attribute of the instance.
        self.BUCKET_NAME = "search-image619" # name of the aws s3 bucket where the images are stored
        self.KEY = "model" #  assigns the string "model" to the KEY attribute of the instance.
        self.ZIP_NAME = "artifacts.tar.gz" #  assigns the string "artifacts.tar.gz" to the ZIP_NAME attribute of the instance.
        self.ZIP_PATHS = [(os.path.join(from_root(), "data", "embeddings", "embeddings.json"), "embeddings.json"), # The first tuple contains the path to the embeddings.json file and the string "embeddings.json".
                          (os.path.join(from_root(), "data", "embeddings", "embeddings.ann"), "embeddings.ann"), # The second tuple contains the path to the embeddings.ann file and the string "embeddings.ann".
                          (os.path.join(from_root(), "model", "finetuned", "model.pth"), "model.pth")] # The third tuple contains the path to the model.pth file and the string "model.pth".

    def get_s3_config(self): #method of the class that returns a dictionary of the instance's attributes.
        return self.__dict__
    




    
