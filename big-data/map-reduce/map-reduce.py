#!/usr/bin/env python3

from bson.code import Code
from pprint import pprint

from init import connect

orders = connect()

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
                if (item.id) {
                    idsDict[item.id] = true;
                } else {
                    idsDict = Object.assign(idsDict, item.idsDict)
                }
                return acc;
            }
            var result = values.reduce(red, { count: 0, purchase: 0 });
            result.idsDict = idsDict;
            return result;
        }
    """)

    result = orders.map_reduce(mapper, reducer, 'purchase-by-gender')
    print('> Purchase by Gender:')
    for doc in result.find():
        print('>>', 'Male:' if doc['_id'] == 'M' else 'Female:')
        value = doc['value']
        unique = len(value['idsDict'])
        purchase = int(value['purchase'])
        avg_by_order = int(purchase / value['count'])
        avg_by_user = int(purchase / unique)

        print(f'\tunique users:\t{unique}')
        print(f'\ttotal purchase:\t{purchase}')
        print(f'\tavg by order:\t{avg_by_order}')
        print(f'\tavg by user:\t{avg_by_user}')

purchase_by_gender()
