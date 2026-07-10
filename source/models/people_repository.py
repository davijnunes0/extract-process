from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.results import DeleteResult, UpdateResult

Document = dict[str, Any]
MongoFilter = dict[str, Any]


class PeopleRepository:
    def __init__(self, db_connection: Database):
        self.__collection_name = "people"
        self.__db_connection = db_connection

    def __get_collection(self) -> Collection:
        return self.__db_connection.get_collection(self.__collection_name)

    @staticmethod
    def __parse_object_id(people_id: str) -> ObjectId:
        try:
            return ObjectId(people_id)
        except (InvalidId, TypeError) as exc:
            raise ValueError("Invalid MongoDB ObjectId.") from exc

    def insert_people(self, people:  Document) -> Document:
        collection = self.__get_collection()
        person_to_insert = dict(people)
        result = collection.insert_one(person_to_insert)
        person_to_insert["_id"] = result.inserted_id
        return person_to_insert

    def insert_many_people(self, list_of_people: list[Document]) -> list[Document]:
        collection = self.__get_collection()
        people_to_insert = [dict(person) for person in list_of_people]
        result = collection.insert_many(people_to_insert)

        for person, inserted_id in zip(people_to_insert, result.inserted_ids):
            person["_id"] = inserted_id

        return people_to_insert

    def select_people_by_id(self, people_id: str) -> Document | None:
        collection = self.__get_collection()
        object_id = self.__parse_object_id(people_id)
        return collection.find_one({"_id": object_id})

    def select_people_by_generic_filter(self, filters: MongoFilter) -> list[Document]:
        collection = self.__get_collection()
        return list(collection.find(filters))

    def select_people_all(self) -> list[Document]:
        collection = self.__get_collection()
        return list(collection.find())

    def update_people_by_id(self, people_id: str, people: Document) -> UpdateResult:
        collection = self.__get_collection()
        object_id = self.__parse_object_id(people_id)
        return collection.update_one({"_id": object_id}, {"$set": people})

    def update_people_by_generic_filter(self, filters: MongoFilter, people: Document) -> UpdateResult:
        collection = self.__get_collection()
        return collection.update_many(filters, {"$set": people})

    def delete_people_by_id(self, people_id: str) -> DeleteResult:
        collection = self.__get_collection()
        object_id = self.__parse_object_id(people_id)
        return collection.delete_one({"_id": object_id})

    def delete_people_by_generic_filter(self, filters: MongoFilter) -> DeleteResult:
        collection = self.__get_collection()
        return collection.delete_many(filters)
