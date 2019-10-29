import json
import numpy as np

def add(a, b):
    return (np.array(a) + np.array(b)).tolist()

HANDLERS = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
}

def lambda_handler(event, context):
    body = json.loads(event['body'])
    func = HANDLERS.get(body['op'])

    if not func:
        return {
            'statusCode': 400,
            'body': 'Unknown operator'
        }

    result = func(np.array(body['a']), np.array(body['b'])).tolist()

    return {
        'statusCode': 200,
        'body': json.dumps({
            'result': result
        })
    }
