


from src.entity.config_entity import DataIngestionConfig## The code imports necessary libraries and modules - DataIngestionConfig and S3Connector from 
from src.utils.storage_handler import S3Connector
from from_root import from_root## the src.entity and src.utils packages, respectively, and the from_root() function from the from_root 
import splitfolders ## module. It also imports the splitfolders module for splitting the data and the built-in os module for 
import os ##handling file and directory operations.


class DataIngestion: ## class DataIngestion

    def __init__(self): ## __init__(self) is the constructor method of the class, which is called when an instance of the class is created.
        self.config = DataIngestionConfig() ##  here in self.config variable we are creating the instance of class DataIngestionConfig

    def download_dir(self): ##The download_dir method downloads data from an S3 bucket using the AWS CLI command aws s3 sync
        """
        params:
        - prefix: pattern to match in s3
        - local: local path to folder in which to place files
        - bucket: s3 bucket with target contents
        - client: initialized s3 client object

        """
        try:
            print("\n====================== Fetching Data ==============================\n") ## printing message Fetching Data
            data_path = os.path.join(from_root(), self.config.RAW, self.config.PREFIX) ## The from_root() function is 
            #a  function that returns the root directory of the project or the current working directory.
            #The self.config.RAW and self.config.PREFIX are instance variables of the DataIngestion class. self.config.RAW 
            # is a string representing the subdirectory within the root directory where the downloaded data will be saved,
            #  while self.config.PREFIX is a string representing the prefix or pattern to match in the S3 bucket.
            os.system(f"aws s3 sync s3://search-image619/images/ {data_path} --no-progress") #The aws s3 sync command is used
            #to synchronize the contents of a local directory with an S3 bucket. The s3://search-image619/images/ is the source
            #  directory in the S3 bucket that will be synchronized with the local directory specified by data_path.
            #The --no-progress option is used to suppress the progress bar that normally appears when executing the aws s3 sync command.
            print("\n====================== Fetching Completed ==========================\n")

        except Exception as e:
            raise e

    def split_data(self):
        """
        This Method is Responsible for splitting.
        The split_data method uses the splitfolders library to split the downloaded data into train, test, and validation 
        sets. Specifically, it calls the ratio() function from the splitfolders library, passing in the path to the input
        directory, the path to the output directory, the random seed, the train-test-validation ratio, and a few other 
        optional parameters. If an exception occurs during this process, it is caught by the try-except block and re-raised.
        :return:
        """
        try:
            splitfolders.ratio(
                input=os.path.join(self.config.RAW, self.config.PREFIX),
                output=self.config.SPLIT,
                seed=self.config.SEED,
                ratio=self.config.RATIO,
                group_prefix=None, move=False
            )
        except Exception as e:
            raise e

    def run_step(self): ## The run_step method is the main entry point for the data ingestion process.
        self.download_dir() ## method to download the data from the S3 bucket to the local file system,
        self.split_data() ##  method to split the downloaded data into train, test, and validation sets.
        return {"Response": "Completed Data Ingestion"} #returns a dictionary object with a key-value pair indicating
        #that the data ingestion process has completed.

### method for initializing a configuration object.
##This is the main block of code that initializes a configuration object for the DataIngestion class, creates the
#  required directories, and executes the data ingestion process.

if __name__ == "__main__":  
    paths = ["data", r"data\raw", r"data\splitted", r"data\embeddings",
             "model", r"model\benchmark", r"model\finetuned"]
    
    """"data" represents the top-level directory where the data will be stored.
    r"data\raw" represents the subdirectory within the data directory where the raw data will be stored.
    r"data\splitted" represents the subdirectory within the data directory where the splitted data will be stored.
    r"data\embeddings" represents the subdirectory within the data directory where the word embeddings will be stored.
    "model" represents the top-level directory where the models will be stored.
    r"model\benchmark" represents the subdirectory within the model directory where the benchmark models will be stored.
    r"model\finetuned" represents the subdirectory within the model directory where the finetuned models will be stored."""
    
    for folder in paths: #This for loop is used to create the directories if they do not already exist.
        path = os.path.join(from_root(), folder) # the os.path.join() function is used to create the full path to
        #the folder by joining the folder name with the root directory of the project obtained using the from_root() function.
        print(path)
        if not os.path.exists(path): #The os.path.exists() function is then used to check if the folder already exists.
            os.mkdir(folder) # If the folder does not exist, then the os.mkdir() function is called to create the folder.

    dc = DataIngestion() ## Instance of DataIngestion class
    print(dc.run_step()) ## calling the run_step method to run the download_dir and split_data methods






