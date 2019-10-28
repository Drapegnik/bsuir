import json
import numpy as np

def add(a, b):
    return (np.array(a) + np.array(b)).tolist()

HANDLERS = {
    '+': add
}

def lambda_handler(event, context):
    body = json.loads(event['body'])
    func = HANDLERS.get(body['op'])

    if not func:
        return {
            'statusCode': 400,
            'body': 'Unknown operator'
        }

    return {
        'statusCode': 200,
        'body': json.dumps({
            'result': func(body['a'], body['b'])
        })
    }
