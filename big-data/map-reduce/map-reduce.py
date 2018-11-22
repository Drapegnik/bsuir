#!/usr/bin/env python3

from bson.code import Code
from pprint import pprint

from init import connect

orders = connect()
sumReducer = Code("""
    function (_, values) {
        return Array.sum(values);
    }
""")

def purchase_by_gender():
    mapper = Code("""
        function () {
            emit(this.Gender, this.Purchase);
        }
    """)

    result = orders.map_reduce(mapper, sumReducer, 'purchase-by-gender')
    print('> purchase_by_gender:')
    for doc in result.find():
        print('\t', end='')
        pprint(doc)

purchase_by_gender()
