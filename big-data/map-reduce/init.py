#!/usr/bin/env python3

import os
import json
import pandas as pd
from pymongo import MongoClient

FILE_PATH = os.path.join(os.path.dirname(__file__), 'data/BlackFriday.csv')

def connect(name = 'orders'):
    client = MongoClient()
    db=client['black-friday']
    collection = db[name]
    return collection

if __name__ == "__main__":
    # read dataset
    data = pd.read_csv(FILE_PATH)
    data_json = json.loads(data.to_json(orient='records'))

    # save to mongo
    orders = connect()
    orders.remove()
    orders.insert(data_json)
    print(f'Successfully insert {len(data)} documents.')
