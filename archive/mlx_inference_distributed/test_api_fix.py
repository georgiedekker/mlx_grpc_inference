import requests
import json

# Test the API
response = requests.post(
    'http://localhost:8100/v1/chat/completions',
    json={
        'model': 'mlx-community/Qwen3-1.7B-8bit',
        'messages': [{'role': 'user', 'content': 'Hello!'}],
        'max_tokens': 10
    }
)

print(f'Status: {response.status_code}')
if response.status_code == 200:
    print('Success! Response:')
    print(json.dumps(response.json(), indent=2))
else:
    print('Error:')
    print(response.text)