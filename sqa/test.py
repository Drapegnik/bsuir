from index import lambda_handler
import json

body = {
    'op': '+',
    'a': [1, 2, 3],
    'b': [3, 2, 1]
}

event_mock = { 'body': json.dumps(body) }

# -> {'statusCode': 200, 'body': '{"result": [4, 4, 4]}'}
print(lambda_handler(event_mock, None))
