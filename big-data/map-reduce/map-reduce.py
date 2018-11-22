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
            emit(this.Gender, {
                purchase: this.Purchase,
                id: this.User_ID,
                count: 1
            });
        }
    """)

    reducer = Code("""
        function (_, values) {
            var idsDict = {};

            function red (acc, item) {
                acc.purchase += item.purchase;
                acc.count += item.count;
                idsDict[item.id] = true;
                return acc;
            }
            var result = values.reduce(red, { count: 0, purchase: 0 });
            result.unique = Object.keys(idsDict).length;
            return result;
        }
    """)

    result = orders.map_reduce(mapper, reducer, 'purchase-by-gender')
    print('> Purchase by Gender:')
    for doc in result.find():
        print('>>', 'Male:' if doc['_id'] == 'M' else 'Female:')
        value = doc['value']
        purchase = value['purchase']
        avg_by_order = purchase / value['count']
        avg_by_user = purchase / value['unique']

        print(f'\ttotal:\t\t{purchase}')
        print(f'\tavg by order:\t{avg_by_order}')
        print(f'\tavg by user:\t{avg_by_user}')

purchase_by_gender()
