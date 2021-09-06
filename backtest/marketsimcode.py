import pandas as pd  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import numpy as np  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import datetime as dt  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
import os  		  	   		     	 

def symbol_to_path(symbol, base_dir=None):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """Return CSV file path given ticker symbol."""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    if base_dir is None:  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        base_dir = './data'		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return os.path.join(base_dir, "{}.csv".format(str(symbol))) 

def get_data(symbols, dates, addSPY=True, colname = 'Adj Close'):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    """Read stock data (adjusted close) for given symbols from CSV files."""  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    df = pd.DataFrame(index=dates)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    if addSPY and 'SPY' not in symbols:  # add SPY for reference, if absent  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        symbols = ['SPY'] + list(symbols) # handles the case where symbols is np array of 'object'  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    for symbol in symbols:  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        df_temp = pd.read_csv(symbol_to_path(symbol), index_col='Date',  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
                parse_dates=True, usecols=['Date', colname], na_values=['nan'])  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        df_temp = df_temp.rename(columns={colname: symbol})  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        df = df.join(df_temp)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        if symbol == 'SPY':  # drop dates SPY did not trade  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
            df = df.dropna(subset=["SPY"])  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return df 	  	   			  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def compute_portvals(orders_file = "./orders/orders.csv", start_val = 1000000, commission=9.95, impact=0.005):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    
    orders = get_orders_data(orders_file)
    prices = get_prices(orders)
    trades = get_trades(orders, prices, commission, impact)
    holdings = get_holdings(trades, start_val)
    portvals = get_values(prices, holdings).sum(axis=1)		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    return pd.DataFrame(index=portvals.index, data=portvals.values)

def get_values(prices, holdings):
    return prices.mul(holdings)

def get_holdings(trades, start_val):
    holdings = trades.copy()
    holdings.iloc[0, holdings.columns.get_loc('Cash')] += start_val
    return holdings.cumsum()

def get_trades(orders, prices, commission=9.95, impact=0.005):
    group = orders.groupby(['Date','Symbol']).sum()
    joined = pivot(group, prices.columns)

    commissions = orders.groupby(['Date']).count().Symbol * commission
    orders = orders.copy()
    orders.loc[orders['Order'] == 'SELL', 'Shares'] *= -1 
    group = orders.groupby(['Date','Symbol']).sum()
    impacts = pivot(group, prices.columns).mul(prices * impact, fill_value=0).sum(axis=1)
    fees = impacts.add(commissions, fill_value=0)

            
    cash = (joined.mul(prices, fill_value=0).sum(axis=1) * -1) - fees
    trades = joined.join(pd.DataFrame(cash[prices.index], columns=['Cash']), how='right')
    return trades.fillna(0)

def pivot(group, columns):
    joined = None

    for col in columns:
        if col != 'Cash':
            ticker = group.query(f'Symbol == "{col}"')
            ticker.index = ticker.index.get_level_values(0)
            ticker.columns = [col]
            joined = joined.join(ticker, how='outer') if not joined is None else ticker
    return joined
 	
def get_prices(orders, spy='$SPX', date_range = None, with_cash=True):
    date_range = date_range if not date_range is None else pd.date_range(orders.index[0], orders.index[-1]) 
    tickers = orders.Symbol.unique() if isinstance(orders, pd.DataFrame) else orders
    prices = get_data(tickers if spy in tickers else np.append(tickers, spy), date_range, False).dropna(subset=[spy])
    if not spy in tickers: prices.drop(columns=spy, inplace=True)
    # fill forward, backward
    prices.ffill(inplace=True)
    prices.bfill(inplace=True)
    # add cash column
    return add_cash_column(prices) if with_cash else prices

def add_cash_column(df):
    df['Cash'] = np.ones(df.shape[0])
    return df

def get_orders_data(orders_file):
    if isinstance(orders_file, pd.DataFrame): return convert_trades_data_frame(orders_file)
    df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan']).sort_index() 
    df.loc[df['Order'] == 'SELL', 'Shares'] *= -1 	 
    return df  	

def convert_trades_data_frame(orders):    
    #orders = orders[orders['Shares'] != 0]
    orders['Order'] = ['SELL' if x < 1 else 'BUY' for x in orders['Shares']]
    orders['Symbol'] = 'JPM'
    orders.index.name = 'Date'
    return orders	

def normalize(df):
    return df / df.ix[0, :]    

def cumulative_return(port_val):
    return (port_val[-1] / port_val[0]) - 1

def average_daily_return(port_val):
    return daily_returns(port_val).mean()

def std_daily_return(port_val):
    return daily_returns(port_val).std()

def sharpe_ratio(port_val, k=252):
    return np.sqrt(k) * average_daily_return(port_val) / std_daily_return(port_val)

def daily_returns(port_val):
    return ((port_val / port_val.shift(1)) - 1)[1:]			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
def test_code():  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # this is a helper function you can use to test your code  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # note that during autograding his function will not be called.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Define input parameters  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    of = "./orders/orders2.csv"  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    sv = 1000000  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Process orders  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    portvals = compute_portvals(orders_file = of, start_val = sv)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    if isinstance(portvals, pd.DataFrame):  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        portvals = portvals[portvals.columns[0]] # just get the first column  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    else:  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
        "warning, code did not return a DataFrame"  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Get portfolio stats  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Here we just fake the data. you should use your code from previous assignments.  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    start_date = dt.datetime(2008,1,1)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    end_date = dt.datetime(2008,6,1)  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = [0.2,0.01,0.02,1.5]  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY = [0.2,0.01,0.02,1.5]  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    # Compare portfolio against $SPX  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Date Range: {start_date} to {end_date}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Sharpe Ratio of Fund: {sharpe_ratio}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Sharpe Ratio of SPY : {sharpe_ratio_SPY}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Cumulative Return of Fund: {cum_ret}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Cumulative Return of SPY : {cum_ret_SPY}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Standard Deviation of Fund: {std_daily_ret}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Standard Deviation of SPY : {std_daily_ret_SPY}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Average Daily Return of Fund: {avg_daily_ret}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Average Daily Return of SPY : {avg_daily_ret_SPY}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print()  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    print(f"Final Portfolio Value: {portvals[-1]}")  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
if __name__ == "__main__":  		  	   		     			  		 			 	 	 		 		 	 		 		 	 		  	 	 			  	 
    test_code()  		  	   		