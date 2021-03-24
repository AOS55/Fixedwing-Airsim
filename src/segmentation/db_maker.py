from pymongo import MongoClient
import os

client = MongoClient('localhost', 27017)

with client:
    db = client.mydatabase
    entrys = db.cars.find()
    print(db.list_collection_names())
    customers = db.customers.find()
    print(customers.next())
    print(customers.next())


# create a database
# dirname = os.path.dirname(__file__)  # get the location of the root directory
# dataset = "tom-showcase"
# dirname = os.path.join(dirname, dataset)
# print(dirname)
# db = client['LookAtMe']
# list_of_db = client.list_database_names()
# print(list_of_db)
