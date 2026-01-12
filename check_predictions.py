import json

data = json.load(open('C:/StockApp/backend/turbomode/data/all_predictions.json'))

print('Sample predictions with full probability breakdowns:')
print('Symbol | Prediction | Confidence | prob_down | prob_neutral | prob_up')
print('-' * 80)

for p in data['predictions'][:20]:
    symbol = p['symbol']
    pred = p['prediction']
    conf = p['confidence']
    pd = p.get('prob_down', 0)
    pn = p.get('prob_neutral', 0)
    pu = p.get('prob_up', 0)

    print(f'{symbol:6s} | {pred:10s} | {conf:.4f}     | {pd:.4f}    | {pn:.4f}       | {pu:.4f}')

print('\nSummary statistics:')
print(f'Total predictions: {len(data["predictions"])}')
print(f'BUY: {data["statistics"]["buy_count"]}')
print(f'SELL: {data["statistics"]["sell_count"]}')
print(f'HOLD: {data["statistics"]["hold_count"]}')
