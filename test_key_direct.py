import requests

key = '66987cfdf999d22.22577826'

print(f'Testing key: {key}')
print(f'Length: {len(key)}')

# Test with eodhistoricaldata.com (main domain)
print('\n1. Testing eodhistoricaldata.com domain:')
r = requests.get(
    'https://eodhistoricaldata.com/api/eod/AAPL.US',
    params={'api_token': key, 'fmt': 'json'},
    timeout=10
)
print(f'   Status: {r.status_code}')
if r.status_code == 200:
    print(f'   SUCCESS!')
else:
    print(f'   Response: {r.text[:200]}')

# Test with eodhd.com (alternate domain)
print('\n2. Testing eodhd.com domain:')
r2 = requests.get(
    'https://eodhd.com/api/eod/AAPL.US',
    params={'api_token': key, 'fmt': 'json'},
    timeout=10
)
print(f'   Status: {r2.status_code}')
if r2.status_code == 200:
    print(f'   SUCCESS!')
else:
    print(f'   Response: {r2.text[:200]}')
