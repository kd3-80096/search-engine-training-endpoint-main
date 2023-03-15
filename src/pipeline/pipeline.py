from src.components.data_ingestion import DataIngestion ## importing the DataIngestion class
from src.components.data_preprocessing import DataPreprocessing ## importing the DataPreprocessing class 
from src.components.embeddings import EmbeddingGenerator, ImageFolder ## importing class EmbeddingGenerator and ImageFolder from embeddings
from src.utils.storage_handler import S3Connector ## importing the S3Connector
from src.components.nearest_neighbours import Annoy ## importing Annoy from nearest_neighbours
from src.components.model import NeuralNet ## importing NeuralNet from model
from src.components.trainer import Trainer ## importing the Trainer from trainer`
from torch.utils.data import DataLoader ## importing the DataLoader from torch
from from_root import from_root ## importing from_root to get root directory in project
from tqdm import tqdm ## importing the tqdm to get dynamically updating progressbar every time a value is requested
import torch ## import torch
import os   ## import os


class Pipeline: ## defining the class as Pipeline which will have methods run various stages of the project

    def __init__(self): ## ## __init__(self) is the constructor method of the class, which is called when an instance of the class is created.
        self.paths = ["data", "data/raw", "data/splitted", "data/embeddings", 
                      "model", "model/benchmark", "model/finetuned"]
        
        """data" represents the top-level directory where the data will be stored.
        r"data\raw" represents the subdirectory within the data directory where the raw data will be stored.
    r"data\splitted" represents the subdirectory within the data directory where the splitted data will be stored.
    r"data\embeddings" represents the subdirectory within the data directory where the word embeddings will be stored.
    "model" represents the top-level directory where the models will be stored.
    r"model\benchmark" represents the subdirectory within the model directory where the benchmark models will be stored.
    r"model\finetuned" represents the subdirectory within the model directory where the finetuned models will be stored."""
    
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  ##The device variable is set to "cuda" if a CUDA-capable GPU is available, and "cpu" otherwise.

    def initiate_data_ingestion(self):
        """ method is responsible for initiating the data ingestion process by creating the necessary directories 
        using the paths variable, and then calling the run_step() method of the DataIngestion class to download and
        split the data."""
        for folder in self.paths:#This for loop is used to create the directories if they do not already exist.
            path = os.path.join(from_root(), folder)# the os.path.join() function is used to create the full path to
        #the folder by joining the folder name with the root directory of the project obtained using the from_root() function.
            if not os.path.exists(path): #The os.path.exists() function is then used to check if the folder already exists.
                os.mkdir(folder)# If the folder does not exist, then the os.mkdir() function is called to create the folder.

        dc = DataIngestion() ## Instance of DataIngestion class
        dc.run_step() ## calling the run_step method to run the download_dir and split_data methods

    @staticmethod
    def initiate_data_preprocessing(): 
        dp = DataPreprocessing() #This method creates an instance of the DataPreprocessing class using dp = DataPreprocessing().
        loaders = dp.run_step() #calling the run_step method to run transformations and create_loaders methods
        return loaders

    @staticmethod
    def initiate_model_architecture():
        return NeuralNet() ## Returns an instance of the NeuralNet class, which defines the architecture of the neural network.

    def initiate_model_training(self, loaders, net): ## method initiating the parameters to train the model
        trainer = Trainer(loaders, self.device, net) ## creating an instance of Trainer class and passing the inputs as loaders,device and net to them
        trainer.train_model() ## calling the train_model method in Trainer class 
        trainer.evaluate(validate=True) ## calling the evaluate method to test model performance
        trainer.save_model_in_pth() ## calling the save_model_in_pth method to save the model in the given path.

    def generate_embeddings(self, loaders, net):
        data = ImageFolder(label_map=loaders["valid_data_loader"][1].class_to_idx)
        dataloader = DataLoader(dataset=data, batch_size=64, shuffle=True)
        embeds = EmbeddingGenerator(model=net, device=self.device)

        for batch, values in tqdm(enumerate(dataloader)):
            img, target, link = values
            print(embeds.run_step(batch, img, target, link))

    @staticmethod
    def create_annoy():
        ann = Annoy()
        ann.run_step()

    @staticmethod
    def push_artifacts():
        connection = S3Connector()
        response = connection.zip_files()
        return response

    def run_pipeline(self):
        self.initiate_data_ingestion()
        loaders = self.initiate_data_preprocessing()
        net = self.initiate_model_architecture()
        self.initiate_model_training(loaders, net)
        self.generate_embeddings(loaders, net)
        self.create_annoy()
        self.push_artifacts()
        return {"Response": "Pipeline Run Complete"}


if __name__ == "__main__":
    image_search = Pipeline()
    image_search.run_pipeline()
