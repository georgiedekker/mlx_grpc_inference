import requests
import json

try:
    response = requests.post(
        'http://localhost:8100/v1/chat/completions',
        json={
            'model': 'mlx-community/Qwen3-1.7B-8bit',
            'messages': [{'role': 'user', 'content': 'Hi!'}],
            'max_tokens': 5
        },
        timeout=30
    )
    
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        print('SUCCESS! Distributed inference working!')
        print(json.dumps(response.json(), indent=2)[:200] + '...')
    else:
        print('Error response:')
        print(response.text[:200])
        
except requests.exceptions.Timeout:
    print('Request timed out - might indicate distributed processing is happening')
except Exception as e:
    print(f'Error: {e}')