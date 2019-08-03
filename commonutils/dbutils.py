import hashlib
from constants import dbconstants
from db import mongobase


def get_dictionary_hash(network_object_dict):
    network_config_string = ''
    for network_value in network_object_dict.values():
        network_config_string = network_config_string + str(network_value) + '_'
    network_config_string = network_config_string[:-1]
    hash_digest = hashlib.sha1(network_config_string.encode())
    return hash_digest


def get_mongo_connection(db_name=dbconstants.DB_NAME, collection_name=dbconstants.COLLECTION_NAME):
    mongo = mongobase.MongoConnector(dbconstants.LOCAL_MONGO_HOSTNAME, dbconstants.LOCAL_MONGO_PORT)
    mongo.set_db(db_name)
    mongo.set_collection(collection_name)
    return mongo


def check_for_duplicate_and_insert(network_dict):
    query_dict = dict({'unique_hash': network_dict['unique_hash']})
    mongo = get_mongo_connection()
    if mongo.check_document(query_dict) is False:
        mongo.insert_document(network_dict)
    mongo.close_connection()
