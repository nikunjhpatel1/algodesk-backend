from flask import Flask, request, jsonify
from flask_cors import CORS
import backtrader as bt
import yfinance as yf
import pandas as pd

app = Flask(__name__)
CORS(app)

class RSIStrategy(bt.Strategy):
    params = (('rsi_period', 14), ('oversold', 30), ('overbought', 70),)
    def __init__(self):
        self.rsi = bt.indicators.RSI(period=self.params.rsi_period)
    def next(self):
        if not self.position and self.rsi < self.params.oversold:
            self.buy()
        elif self.position and self.rsi > self.params.overbought:
            self.sell()

class MACDStrategy(bt.Strategy):
    def __init__(self):
        self.macd = bt.indicators.MACD()
        self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)
    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
        elif self.position and self.crossover < 0:
            self.sell()

class SupertrendStrategy(bt.Strategy):
    params = (('period', 10),)
    def __init__(self):
        self.ema = bt.indicators.EMA(period=self.params.period)
    def next(self):
        if not self.position and self.data.close[0] > self.ema[0]:
            self.buy()
        elif self.position and self.data.close[0] < self.ema[0]:
            self.sell()

class BollingerStrategy(bt.Strategy):
    def __init__(self):
        self.bb = bt.indicators.BollingerBands()
    def next(self):
        if not self.position and self.data.close[0] < self.bb.lines.bot[0]:
            self.buy()
        elif self.position and self.data.close[0] > self.bb.lines.top[0]:
            self.sell()

STRATEGIES = {
    'RSI Reversal':     RSIStrategy,
    'MACD Crossover':   MACDStrategy,
    'Supertrend':       SupertrendStrategy,
    'Bollinger Bounce': BollingerStrategy,
}

@app.route('/backtest', methods=['POST'])
def run_backtest():
    try:
        body      = request.json
        symbol    = body.get('symbol', 'RELIANCE.NS')
        strategy  = body.get('strategy', 'RSI Reversal')
        from_date = body.get('from_date', '2023-01-01')
        to_date   = body.get('to_date',   '2023-12-31')
        capital   = float(body.get('capital', 100000))

        df = yf.download(symbol, start=from_date, end=to_date, auto_adjust=True, progress=False)
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        if df.empty:
            return jsonify({'error': 'No data found for symbol'}), 400

        cerebro = bt.Cerebro()
        cerebro.addstrategy(STRATEGIES[strategy])
        data = bt.feeds.PandasData(dataname=df)
        cerebro.adddata(data)
        cerebro.broker.setcash(capital)
        cerebro.broker.setcommission(commission=0.001)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio,   _name='sharpe')
        cerebro.addanalyzer(bt.analyzers.DrawDown,      _name='drawdown')
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

        results     = cerebro.run()
        strat       = results[0]
        final_value = cerebro.broker.getvalue()
        net_return  = ((final_value - capital) / capital) * 100

        trade_analysis = strat.analyzers.trades.get_analysis()
        total_trades = int(trade_analysis.get('total', {}).get('closed', 0))
        won          = int(trade_analysis.get('won',   {}).get('total',  0))
        lost         = int(trade_analysis.get('lost',  {}).get('total',  0))
        win_rate     = round((won / total_trades * 100) if total_trades > 0 else 0, 1)

        sharpe   = strat.analyzers.sharpe.get_analysis().get('sharperatio', 0) or 0
        drawdown = strat.analyzers.drawdown.get_analysis().get('max', {}).get('drawdown', 0)

        return jsonify({
            'net_return':   round(net_return, 2),
            'final_value':  round(final_value, 2),
            'total_trades': total_trades,
            'won':          won,
            'lost':         lost,
            'win_rate':     win_rate,
            'sharpe':       round(float(sharpe), 2),
            'max_drawdown': round(float(drawdown), 2),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return jsonify({'status': 'AlgoDesk Backtest Engine running!'})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
