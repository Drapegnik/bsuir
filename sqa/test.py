from index import lambda_handler
import json

body = {
    'op': '*',
    'a': [1, 2, 3],
    'b': 10
}

event_mock = { 'body': json.dumps(body) }

# -> {'statusCode': 200, 'body': '{"result": [10, 20, 30]}'}
print(lambda_handler(event_mock, None))
