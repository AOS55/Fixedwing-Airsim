from pymongo import MongoClient
import os

client = MongoClient('localhost', 27017)

# create a database
dirname = os.path.dirname(__file__)  # get the location of the root directory
dataset = "tom-showcase"
dirname = os.path.join(dirname, dataset)
print(dirname)
db = client['LookAtMe']
list_of_db = client.list_database_names()
print(list_of_db)
