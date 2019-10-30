import json
import numpy as np
from aws_xray_sdk.core import xray_recorder


@xray_recorder.capture('add')
def add(a, b):
    return a + b


@xray_recorder.capture('sub')
def sub(a, b):
    return a - b


@xray_recorder.capture('mul')
def mul(a, b):
    return a * b


HANDLERS = {
    '+': add,
    '-': sub,
    '*': mul,
}


def lambda_handler(event, context):
    try:
        body = json.loads(event['body']) if type(event['body']) is str else event['body']
        func = HANDLERS.get(body['op'])

        if not func:
            return {
                'statusCode': 400,
                'body': 'Unknown operator'
            }

        with xray_recorder.in_segment('computation') as segment:
            a = np.array(body['a'])
            b = np.array(body['b'])
            result = func(a, b).tolist()
            segment.put_annotation('size', f'{a.size * b.size}')

        return {
            'statusCode': 200,
            'body': json.dumps({
                'result': result
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'input': event['body']
            })
        }
