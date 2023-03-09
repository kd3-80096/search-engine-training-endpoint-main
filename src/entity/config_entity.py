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
        self.BUCKET: str = "image-database-system-01" # assigns the string "search-image619" to the BUCKET attribute of the instance
        self.SEED: int = 1337  #assigns the integer 1337 to the SEED attribute of the instance.
        self.RATIO: tuple = (0.8, 0.1, 0.1)  # to the RATIO attribute of the instance.

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
        self.BUCKET: str = "image-database-system-01"
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


class s3Config:
    def __init__(self):
        self.ACCESS_KEY_ID = os.environ["ACCESS_KEY_ID"]
        self.SECRET_KEY = os.environ["AWS_SECRET_KEY"]
        self.REGION_NAME = "ap-south-1"
        self.BUCKET_NAME = "image-database-system-01"
        self.KEY = "model"
        self.ZIP_NAME = "artifacts.tar.gz"
        self.ZIP_PATHS = [(os.path.join(from_root(), "data", "embeddings", "embeddings.json"), "embeddings.json"),
                          (os.path.join(from_root(), "data", "embeddings", "embeddings.ann"), "embeddings.ann"),
                          (os.path.join(from_root(), "model", "finetuned", "model.pth"), "model.pth")]

    def get_s3_config(self):
        return self.__dict__
