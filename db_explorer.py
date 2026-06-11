import pymongo
import json
from bson import ObjectId
from datetime import datetime

class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

def main():
    client = pymongo.MongoClient("mongodb://127.0.0.1:27017")
    db = client["ECMS_MAY03_COPY"]
    
    collections = sorted(db.list_collection_names())
    print("All collections:")
    for col in collections:
        if "template" in col:
            count = db[col].count_documents({})
            print(f"- {col} ({count} docs)")
            
    # Print escalationLevel values from task collection
    task_col = None
    for col in collections:
        if "task_template" in col:
            task_col = col
            break
            
    if task_col:
        print(f"\nValues of escalationLevel in {task_col}:")
        cursor = db[task_col].find({}, {"escalationLevel": 1, "taskId": 1})
        for doc in cursor:
            print(f"  Task {doc.get('taskId')}: escalationLevel = {doc.get('escalationLevel')}")
            
if __name__ == "__main__":
    main()
