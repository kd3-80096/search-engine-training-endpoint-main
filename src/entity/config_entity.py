from from_root import from_root ## importing the library from_root to get the root directory of the project
from dotenv import load_dotenv ## importing the load_dotenv that loads all variables in .env file
import os ## importing os for system operations


class DatabaseConfig: ## class name is DatabaseConfig

    def __init__(self): #__init__(self) is the constructor method of the class, which is called when an instance of the class is created.
        # Load environment variables from .env file
        load_dotenv() # method that loads  environment variables from a .env file.
        self.USERNAME: str = os.getenv("ATLAS_CLUSTER_USERNAME") #os.environ["DATABASE_USERNAME"] assigns the value of 
        #the environment variable DATABASE_USERNAME to the USERNAME attribute of the instance.
        self.PASSWORD: str = os.getenv("DATABASE_PASSWORD")
        self.URL: str = os.getenv("ATLAS_CLUSTER_PASSWORD") #os.getenv("MONGODB_URL_KEY") assigns the value of the 
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


class DataPreprocessingConfig: ## class name is DataPreprocessingConfig
    def __init__(self):#__init__(self) is the constructor method of the class, which is called when an instance of the class is created.
        self.BATCH_SIZE = 32 ## 32 images will be passed in one batch size
        self.IMAGE_SIZE = 256 ## image size is 256
        self.TRAIN_DATA_PATH = os.path.join(from_root(), "data", "splitted", "train") ##TRAIN_DATA_PATH is stored in self.TRAIN_DATA_PATH 
        self.TEST_DATA_PATH = os.path.join(from_root(), "data", "splitted", "test") # TEST_DATA_PATH is stored in self.TEST_DATA_PATH
        self.VALID_DATA_PATH = os.path.join(from_root(), "data", "splitted", "valid") #VALID_DATA_PATH is stored in self.VALID_DATA_PATH

    def get_data_preprocessing_config(self):
        return self.__dict__ #  method of the class that returns a dictionary of the instance's attributes.


class ModelConfig: ## class name is ModelConfig which defines the inputs to the model creation
    def __init__(self):
        self.LABEL = 101 ## total number of labels of images is 101
        self.STORE_PATH = os.path.join(from_root(), "model", "benchmark") ## path where the model will be stored is 
        self.REPOSITORY = 'pytorch/vision:v0.10.0' ##set the REPOSITORY variable to the PyTorch vision library version v0.10.0.
        self.BASEMODEL = 'resnet18'  ## algorithm used is resnet-18
        self.PRETRAINED = True ## pretraining is set to true

    def get_model_config(self):
        return self.__dict__ ## returning the attributes and its values in dictionary format.


class TrainerConfig: ## TrainerConfig class, which has three attributes initialized in the constructor: MODEL_STORE_PATH, EPOCHS, and Evaluation.
    def __init__(self): 
        self.MODEL_STORE_PATH = os.path.join(from_root(), "model", "finetuned", "model.pth") ##  used to join the 
#root directory of the project (returned by the from_root() function) with the directory "model/finetuned" and the file "model.pth". 
        self.EPOCHS = 2 ## 2 number of epochs the model will be trained for during the fine-tuning process.
        self.Evaluation = True ##True boolean attribute that specifies whether the model should be evaluated on a validation set after each epoch of training.

    def get_trainer_config(self):
        return self.__dict__ ## returning the attributes and its values in dictionary format.


class ImageFolderConfig: ## class name is ImageFolderConfig
    """ class is used to define a set of configurations related to the image folder data, such as the directory path,
   image size, label map, S3 bucket name, and S3 link format."""   
    def __init__(self): ##__init__(self) is the constructor method of the class, which is called when an instance of the class is created.
        self.ROOT_DIR = os.path.join(from_root(), "data", "raw", "images") # The path where the raw images will be stored
        self.IMAGE_SIZE = 256 ## size of image will be 256
        self.LABEL_MAP = {} ##dictionary that maps class names to class indices.
        self.BUCKET: str = "search-image619" ##  string representing the name of the S3 bucket where the images are stored.
        self.S3_LINK = "https://{0}.s3.ap-south-1.amazonaws.com/images/{1}/{2}" ## string representing the format of the link to an image in the S3 bucket.
        

    def get_image_folder_config(self):
        return self.__dict__ #method returns a dictionary containing all the attributes of the ImageFolderConfig instance


class EmbeddingsConfig: #EmbeddingsConfig class has the path of the Model stored in directory
    def __init__(self):
        self.MODEL_STORE_PATH = os.path.join(from_root(), "model", "finetuned", "model.pth") # used to join the 
#root directory of the project (returned by the from_root() function) with the directory "model/finetuned" and the file "model.pth". 

    def get_embeddings_config(self):
        return self.__dict__ ## method returns a dictionary containing all attributes of the EmbeddingsConfig instance.


class AnnoyConfig:
    def __init__(self):
        self.EMBEDDING_STORE_PATH = os.path.join(from_root(), "data", "embeddings", "embeddings.ann")

    def get_annoy_config(self):
        return self.__dict__


class s3Config: ## name of class is s3Config

    def __init__(self): ## constructor method of the class, which is called when an instance of the class is created.
        self.ACCESS_KEY_ID = os.getenv["ACCESS_KEY_ID"] # os.environ["ACCESS_KEY_ID"] assigns the value of the 
        #environment variable ACCESS_KEY_ID to the ACCESS_KEY_ID attribute of the instance.
        self.SECRET_KEY = os.getenv["AWS_SECRET_KEY"] # 
        self.REGION_NAME = "ap-south-1" # assigns the string "ap-south-1" to the REGION_NAME attribute of the instance.
        self.BUCKET_NAME = "search-image619" # name of the aws s3 bucket where the images are stored
        self.KEY = "model" #  assigns the string "model" to the KEY attribute of the instance.
        self.ZIP_NAME = "artifacts.tar.gz" #  assigns the string "artifacts.tar.gz" to the ZIP_NAME attribute of the instance.
        self.ZIP_PATHS = [(os.path.join(from_root(), "data", "embeddings", "embeddings.json"), "embeddings.json"), # The first tuple contains the path to the embeddings.json file and the string "embeddings.json".
                          (os.path.join(from_root(), "data", "embeddings", "embeddings.ann"), "embeddings.ann"), # The second tuple contains the path to the embeddings.ann file and the string "embeddings.ann".
                          (os.path.join(from_root(), "model", "finetuned", "model.pth"), "model.pth")] # The third tuple contains the path to the model.pth file and the string "model.pth".

    def get_s3_config(self): #method of the class that returns a dictionary of the instance's attributes.
        return self.__dict__
    




    
