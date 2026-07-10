from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

URI = os.getenv("MONGO_URI")
NAME_DB = os.getenv("MONGO_DATABASE")


class DbConnectionHandler:
    def __init__(self):
        self.__connection_string = URI
        self.__client = None
        self.__db_connection = None

    def connect_to_db(self):
        if not self.__connection_string:
            raise ValueError("MONGO_URI is not configured.")

        if not NAME_DB:
            raise ValueError("MONGO_DATABASE is not configured.")

        self.__client = MongoClient(self.__connection_string)
        self.__db_connection = self.__client[NAME_DB]
        return self.__db_connection

    def get_db_connection(self):
        return self.__db_connection

    def get_db_client(self):
        return self.__client
