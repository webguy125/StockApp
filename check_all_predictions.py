import json

data = json.load(open('C:/StockApp/backend/turbomode/data/all_predictions.json'))

print(f'Total: {data["total"]}')
print(f'BUY: {data["statistics"]["buy_count"]}')
print(f'SELL: {data["statistics"]["sell_count"]}')
print(f'HOLD: {data["statistics"]["hold_count"]}')
print(f'Timestamp: {data["timestamp"]}')
print('\nAll predictions:')
print('='*90)

for i, p in enumerate(data['predictions'], 1):
    symbol = p['symbol']
    pred = p['prediction']
    conf = p['confidence']
    pd = p.get('prob_down', 0)
    pn = p.get('prob_neutral', 0)
    pu = p.get('prob_up', 0)

    print(f'{i:2d}. {symbol:6s} | {pred:4s} | conf:{conf:.1%} | down:{pd:.1%} neutral:{pn:.1%} up:{pu:.1%}')
