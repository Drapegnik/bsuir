#!/usr/bin/env python3

import matplotlib.pyplot as plt

from bson.code import Code
from pprint import pprint
from time import time

from init import connect

orders = connect()
labels = []

def log(legend, label, value):
    print(f'\t{legend}:\t\t{value}')

def draw(legend, data):
    labels_ = []
    for i, item in enumerate(data):
        labels_.append(f'{labels[i]} ({item})')

    plt.clf()
    plt.pie(data, labels=labels_, colors=['lightcoral', 'lightskyblue'], autopct='%1.1f%%')
    filename = legend.replace(' ', '-')
    plt.savefig(f'images/{filename}.png')

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

    start_time = time()
    result = orders.map_reduce(mapper, reducer, 'purchase-by-gender')
    elapsed_time = time() - start_time
    print('> Purchase by Gender:')
    users = []
    totals = []
    avgs_by_order = []
    avgs_by_user = []

    for doc in result.find():
        label = 'Male' if doc['_id'] == 'M' else 'Female'
        labels.append(label)
        value = doc['value']
        unique = len(value['idsDict'])
        purchase = int(value['purchase'])
        avg_by_order = int(purchase / value['count'])
        avg_by_user = int(purchase / unique)

        users.append(unique)
        totals.append(purchase)
        avgs_by_order.append(avg_by_order)
        avgs_by_user.append(avg_by_user)

        print(f'>> {label}:')
        log('unique users', label, unique)
        log('total purchase', label, purchase)
        log('avg by order', label, avg_by_order)
        log('avg by user', label, avg_by_user)

    print(f'> Elapsed Time:\t{elapsed_time:.2f}s')
    draw('users', users)
    draw('total purchase', totals)
    draw('avg by order', avgs_by_order)
    draw('avg by user', avgs_by_user)

purchase_by_gender()
