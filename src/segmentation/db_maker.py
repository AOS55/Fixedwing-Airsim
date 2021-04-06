from pymongo import MongoClient
import gridfs
import json
import os

"""
Database creation steps:
1. Check to see if the database exists, database_name = "segmentation"
        If it does not create it
2. Check to see if the collection exists, collection_name = dataset_name
        If it does not create it
3. Create a document with the combination of the flight_data, images, metadata better to store png in filesystem and call them
4. Insert the document into the database
"""


class NoSQLDB:

    def __init__(self,
                client: MongoClient,
                db_name: str = 'segmentation',
                collection_name: str = 'validation-dataset'):
        self.client = client
        self.db_name = db_name
        self.db = self.create_database()
        self.collection_name = collection_name
        self.collection = self.create_colleciton()

    def create_database(self):
        """
        Check if a database exists and return the connection if it exists

        :return: a connection the the database
        """
        db_names = self.client.list_database_names()
        if self.db_name not in db_names:
            print(f'{self.db_name} not on cluster, this will be created')
        return self.client[self.db_name]

    def create_colleciton(self):
        """
        Check if a collection exists in a database and return it if it does

        :return: a connection the collection
        """
        collection_names = self.db.list_collection_names()
        if self.collection_name not in collection_names:
            print(f'{self.collection_name} not in database, this will be created')
        return self.db[self.collection_name]
    
    def store_image(self, image_loc: str):
        """
        Stores an image as a binary gridfs file on the database
        ...

        Aim to not put the image on the database as this is inefficient,
        just pass the path to the image on the file system.
        """
        fs = gridfs.GridFS(database=self.db, collection="myCollection")
        with open(image_loc, 'rb') as f:
            contents = f.read()
        fs.put(contents, filename="file", bar="baz")

    def create_document_json(self, file):
        """
        Generates a single JSON file from a specified JSON location
        """
        with open(file) as f:
            file_data = json.load(f)
        self.collection.insert_one(file_data)

    def create_group_json(self, path):
        """
        Generates a collection of documents from a JSON dir
        """
        for json_file in os.listdir(path):
            cur_json = os.path.join(path, json_file)
            self.create_document_json(cur_json)


dataset = 'meta-validation'
dirname = os.path.dirname(__file__)  # get the location of the root directory
dirname = os.path.join(dirname, '../..')  # move out of segmentation source directory
dirname = os.path.join(dirname, 'data/segmentation-datasets')  # go into segmentation-dataset dir
metadata_dir = os.path.join(dirname + '/' + dataset + '/metadata')  # combine to go to metadata dir
my_client = MongoClient('localhost', 27017)  # connect to database
my_db = 'segmentation'
edit_db = NoSQLDB(my_client, collection_name="input-validation")
my_metadata = os.path.join(metadata_dir, '10.json')
edit_db.create_group_json(metadata_dir)


# test_image = "C:/Users/quessy/Documents/Year_5_PhD/Simulation/python-client/data/segmentation-datasets/fd-validation/images/0.png"
# edit_db.store_image(test_image)



# with client:
#     db = client.mydatabase
#     entrys = db.cars.find()
#     print(db.list_collection_names())
#     customers = db.customers.find()
#     print(customers.next())
#     print(customers.next())


# create a database
# dirname = os.path.dirname(__file__)  # get the location of the root directory
# dataset = "tom-showcase"
# dirname = os.path.join(dirname, dataset)
# print(dirname)
# db = client['LookAtMe']
# list_of_db = client.list_database_names()
# print(list_of_db)
