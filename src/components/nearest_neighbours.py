from src.utils.database_handler import MongoDBClient #importing the mongodb client that 
from src.entity.config_entity import AnnoyConfig ## importing the AnnoyConfig class 
from annoy import AnnoyIndex ##he AnnoyIndex class is the core class of the Annoy library and is used to build and 
#query Annoy indexes. An Annoy index is a data structure that allows for efficient nearest neighbor searches in 
# high-dimensional spaces by partitioning the space into small hyper-rectangles (known as "nodes") and indexing the
# points based on the nodes they belong to.
from typing_extensions import Literal  #The Literal class is useful for enforcing strict type constraints in your code
from tqdm import tqdm
import json

"""code is trying to build an Annoy index from image embeddings stored in a MongoDB database and save it to a file, 
which can be used for efficient nearest neighbor searches."""

class CustomAnnoy(AnnoyIndex): #class called CustomAnnoy that inherits from the AnnoyIndex class.
    def __init__(self, f: int, metric: Literal["angular", "euclidean", "manhattan", "hamming", "dot"]):
#  method is the constructor for the CustomAnnoy class. It takes two arguments: f, which is an integer, and metric,
#  which is a Literal type that can only take on the values "angular", "euclidean", "manhattan", "hamming", or "dot". 
        super().__init__(f, metric) #It calls the constructor of the AnnoyIndex class with these arguments
        self.label = [] #initializes an empty list called self.label.

    # noinspection PyMethodOverriding
    def add_item(self, i: int, vector, label: str) -> None: #This method overrides the add_item method of the AnnoyIndex class
#takes three arguments: i, which is an integer, vector, which can be any type, and label, which is a string        
        super().add_item(i, vector) # calls the add_item method of the parent class with the first two arguments
        self.label.append(label) # appends the label argument to the self.label list.

    def get_nns_by_vector( #  method overrides the get_nns_by_vector method of the AnnoyIndex class.
            self, vector, n: int, search_k: int = ..., include_distances: Literal[False] = ...):
#It takes three arguments: vector, which can be any type, n which is an integer, and search_k and include_distances, which are optional arguments with default values.
        indexes = super().get_nns_by_vector(vector, n, search_k, include_distances) #calls the get_nns_by_vector method of the parent class with these arguments,
        labels = [self.label[link] for link in indexes] # then creates a new list called labels by iterating over
        #the indexes list and looking up the corresponding label value in the self.label list.
        return labels #returns the labels list.

    def load(self, fn: str, prefault: bool = ...): #method overrides the load method of the AnnoyIndex class.
    #takes two arguments: fn, which is a string, and prefault, which is an optional boolean argument with a default value
        super().load(fn) #calls the load() method of the superclass AnnoyIndex with the argument fn.
        path = fn.replace(".ann", ".json") #line creates a new string path by replacing the extension .ann in the fn argument with .json.
        self.label = json.load(open(path, "r")) #loads the JSON file at the path and assigns the result to the label instance variable of the CustomAnnoy` class.

    def save(self, fn: str, prefault: bool = ...): 
        """another overides method of the CustomAnnoy class. It takes two arguments, a string fn representing the file name to 
        save and an optional boolean prefault. It saves the index to the specified file and also saves the
        corresponding labels to a JSON file."""
        super().save(fn)#calls the save() method of the superclass AnnoyIndex with the argument fn.
        path = fn.replace(".ann", ".json") #creates a new string path by replacing the extension .ann in the fn argument with .json.
        json.dump(self.label, open(path, "w")) #saves the label instance variable of the CustomAnnoy class to a JSON file at the path path.


class Annoy(object): ##Defines a class Annoy as a subclass of the object class.
    def __init__(self): 
        self.config = AnnoyConfig() #self.config: an instance of the AnnoyConfig class
        self.mongo = MongoDBClient() # self.mongo: an instance of the MongoDBClient class
        self.result = self.mongo.get_collection_documents()["Info"] # a list of documents retrieved from a MongoDB collection

    def build_annoy_format(self): # method creates an instance of the CustomAnnoy class, adds items to it using data from self.result, and saves the index to a file.
        Ann = CustomAnnoy(256, 'euclidean') #Creates an instance of the CustomAnnoy class with 256 dimensions and the 'euclidean' distance metric.
        print("Creating Ann for predictions : ")
        for i, record in tqdm(enumerate(self.result), total=8677): #for loop that iterates over the documents retrieved from a MongoDB collection stored in the self.result list.
            Ann.add_item(i, record["images"], record["s3_link"]) #Adds an item to the CustomAnnoy index with an 
            #index i, a vector record["images"], and a label record["s3_link"]

        Ann.build(100) #Builds the CustomAnnoy index using 100 trees.
        Ann.save(self.config.EMBEDDING_STORE_PATH) # Saves the CustomAnnoy index to a file specified by EMBEDDING_STORE_PATH in the AnnoyConfig class.
        return True

    def run_step(self): #method calls the build_annoy_format method.
        self.build_annoy_format()


if __name__ == "__main__":
    ann = Annoy()  ## creating the instance of the annoy class 
    ann.run_step() ## running the run_step method through the object of the annoy class 







