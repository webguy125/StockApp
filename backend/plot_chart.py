import pandas as pd
import plotly.graph_objects as go
import os

def plot_candlestick(symbol, outdir="data"):
    filename = os.path.join(outdir, f"{symbol}.csv")

    if not os.path.exists(filename):
        print(f"No CSV found for {symbol}. Run the fetch script first.")
        return

    df = pd.read_csv(filename, parse_dates=["Date"])

    fig = go.Figure(data=[go.Candlestick(
        x=df["Date"],
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"]
    )])

    fig.update_layout(
        title=f"{symbol} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    fig.show()

if __name__ == "__main__":
    symbol = input("Enter a stock symbol to plot: ").strip().upper()
    if symbol:
        plot_candlestick(symbol)
    else:
        print("No symbol entered. Exiting.")