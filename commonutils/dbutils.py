import hashlib
from constants import dbconstants, model_constants
from db import mongobase


def get_dictionary_hash(preprocess_object_dict):
    hash_config_string = ''
    for hash_column in model_constants.HASH_GENERATION_COLUMN_LIST:
        hash_config_string = hash_config_string + str(preprocess_object_dict[hash_column]) + '_'
    hash_config_string = hash_config_string[:-1]
    hash_digest = hashlib.sha1(hash_config_string.encode())
    return hash_digest.hexdigest()


def get_mongo_connection(db_name=dbconstants.DB_NAME, collection_name=dbconstants.COLLECTION_NAME):
    mongo = mongobase.MongoConnector(dbconstants.LOCAL_MONGO_HOSTNAME, dbconstants.LOCAL_MONGO_PORT)
    mongo.set_db(db_name)
    mongo.set_collection(collection_name)
    return mongo


def check_for_duplicate_and_insert(preprocess_dict):
    query_dict = dict({'unique_hash': preprocess_dict['unique_hash']})
    mongo = get_mongo_connection()
    if mongo.check_document(query_dict) is False:
        mongo.insert_document(preprocess_dict)
    mongo.close_connection()
