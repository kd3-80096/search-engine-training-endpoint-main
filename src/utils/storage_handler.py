from src.entity.config_entity import s3Config ## importing the s3Config class from config_entity
import tarfile ## imports the tarfile module for working with tar archives
from boto3 import Session #the Session class from the boto3 module for interacting with Amazon Web Services (AWS) services.
import os # os module for working with the operating system.

"""code defines a class called S3Connector which is used to interact with an Amazon S3 bucket. It initializes the 
class with the AWS access key, secret key, region, bucket name and other configurations. The class has two methods 
- zip_files and pull_artifacts."""

class S3Connector(object): ## class named S3Connector
    def __init__(self):
        self.config = s3Config() ##creating instance of the s3Config class
        self.session = Session(aws_access_key_id=self.config.ACCESS_KEY_ID, #The aws_access_key_id, aws_secret_access_key,
                               
                               aws_secret_access_key=self.config.SECRET_KEY,# and region_name are passed as arguments to the Session constructor,
                               region_name=self.config.REGION_NAME)
        ##Session object is used to create an S3 client and resource instance.
        self.client = self.session.client("s3")
        self.s3 = self.session.resource("s3")
        self.bucket = self.s3.Bucket(self.config.BUCKET_NAME) #self.bucket attribute is set to an instance of the 
        #Bucket class, which represents the S3 bucket specified in the s3Config instance.

    def zip_files(self):
        """zip_files method creates a new tar archive and adds files from the specified paths. The archive is then uploaded to the S3 bucket."""
        folder = tarfile.open(self.config.ZIP_NAME, "w:gz") # line creates a new tar archive file with the name specified in self.config.ZIP_NAME and opens it in write mode with gzip compression.
        print(folder) ## prints the file_name
        for path, name in self.config.ZIP_PATHS: # line loops through a list of tuples self.config.ZIP_PATHS which 
        #contain the path to the file to be added to the archive and the name it will have inside the archive.
            folder.add(path, name) # line adds the file at path to the archive and gives it the  name inside the archive.
        folder.close() #line closes the tar archive file.

        self.s3.meta.client.upload_file(self.config.ZIP_NAME, self.config.BUCKET_NAME, # line uploads the tar 
#archive file self.config.ZIP_NAME to the S3 bucket specified in self.config.BUCKET_NAME, with the key (path within the bucket)
#  specified in self.config.KEY, and the same name as the original file.
                                        f'{self.config.KEY}/{self.config.ZIP_NAME}')
        os.remove(self.config.ZIP_NAME) # line removes the tar archive file from the local filesystem.

    def pull_artifacts(self):
        """pull_artifacts method downloads the tar archive from the S3 bucket, extracts it to the current directory, and removes the archive file."""
        self.bucket.download_file(f'{self.config.KEY}/{self.config.ZIP_NAME}', self.config.ZIP_NAME) #downloads the
#tar archive from the S3 bucket with the specified key and file name and saves it to the local file system with the same file name.
        folder = tarfile.open(self.config.ZIP_NAME) #opens the downloaded tar archive file using the tarfile module.
        folder.extractall() # extracts all the files in the tar archive to the current directory.
        folder.close() #closes the tar archive file.
        os.remove(self.config.ZIP_NAME) # removes the downloaded tar archive file from the local file system to avoid taking up unnecessary disk space.


if __name__ == "__main__":
    connection = S3Connector()  ## creating instance of S3Connector class 
    # connection.zip_files()
    # connection.pull_artifacts()
