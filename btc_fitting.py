import numpy as np
import pandas as pd
import yfinance as yf
import lmfit as lm
from matplotlib_profile import *
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogFormatter
from sklearn.linear_model import QuantileRegressor, LinearRegression


################ Importing data set from file and API ################

genesis_block_date = np.datetime64('2009-01-03')
btc_halving_dates=['2012-11-28','2016-07-09','2020-05-11','2024-04-19','2028-04-17'] # 2028 is predicted
avg_days_per_year = 365.25

def date_to_daygen(date, genesis_block_date):
    '''convert a date into the number of days after the first BTC block, aka genesis block'''
    return (pd.to_datetime(date)-genesis_block_date).days

def daygen_to_date(daygen, genesis_block_date):
    '''convert a date into the number of days after the first BTC block, aka genesis block'''
    return daygen + genesis_block_date

def adjust_inflation_since_gen(x, m2_growth_rate):
    '''adjust price to genesis block prices by taking into account money supply growth'''
    return x['Price'] * (1 + m2_growth_rate)**(-x['DAY_GEN']/avg_days_per_year)

def get_real_price(x, m2_growth_rate):
    '''opposite of adjust_inflation_since_gen: re-adjust price to real world price'''
    return x['Price_GEN'] * (1 + m2_growth_rate)**(x['DAY_GEN']/avg_days_per_year)

def import_btc_price(hist_price_file, add_recent_price=True, recent_start_date = '2024-09-25', bypass_yfinance=True):
    '''Import daily price of BTC from a data file, and optionally from API'''
    # Import historical data
    btc_history=pd.read_csv(hist_price_file)
    btc_history.index=pd.to_datetime(btc_history.Start)
    btc_history.index.rename('Date', inplace=True)
    btc_history.rename(columns={'Close':'Price'}, inplace=True)
    # btc_history.rename({'index':'Date','Close':'Price'}, inplace=True)
    if add_recent_price: # Add recent data with yahoo finance API
        # to by-pass yfinance download limitation
        if bypass_yfinance:
            from curl_cffi import requests
            session = requests.Session(impersonate="chrome")
            ticker = yf.Ticker('...', session=session)
        btc_price_recent = yf.download("BTC-USD",start=recent_start_date, interval='1d')
        # Gets rid of regarded pandas MultiIndex
        btc_price_recent = pd.DataFrame({'Date':btc_price_recent.index.values, 'Price':btc_price_recent.Close['BTC-USD'].values}).set_index('Date')
        btc_price= pd.concat([btc_history, btc_price_recent])
    else:
        btc_price = btc_history[['Price']].copy()
    btc_price['DAY_GEN'] = btc_price.apply(lambda x:(x.name - genesis_block_date).days, axis=1)
    btc_price.sort_index(inplace=True)
    return btc_price

def find_halving_infos(btc_price):
    btc_halving_daygen = np.array([date_to_daygen(h, genesis_block_date) for h in btc_halving_dates])
    halving_mean_duration = np.diff(btc_halving_daygen).mean()
    # Find cycle tops
    peak_cycle_idx=[]
    for  d in range(len(btc_halving_dates)-2):
        start_cycle = btc_halving_dates[d]
        end_cycle = np.datetime64(btc_halving_dates[d])+np.timedelta64(1000,'D')
        btc_cycle_d=btc_price[(btc_price.index>start_cycle)&(btc_price.index<end_cycle)]
        peak_cycle_idx.append(btc_cycle_d.Price.idxmax())
    return btc_halving_daygen, halving_mean_duration, peak_cycle_idx


################ PL support fitting ################

class PL:
    def __init__(self, res_lin):
        '''take the result of a linear fit in log-log, returns the corresponding power-law function'''
        self.amplitude = np.exp(res_lin.best_values['intercept'])
        self.exponent = res_lin.best_values['slope']
    def eval(self, x):
        return self.amplitude * (x ** self.exponent)

def fit_PL_loglog(df, quantile=0.1, date_fit_start=None, date_fit_end=None, yaxis='Price', plot=True):
    '''fit support line using quantile regression - keeps lower values, excludes high outliers
    quantile: 0.1 = 10th percentile (lower support), 0.05 = 5th percentile (stricter)
    fit_lmfit: if True, also fit lmfit LinearModel to inlier points only
    '''
    # Select and convert values to log-log space
    if date_fit_start: df=df[df.index>date_fit_start]
    if date_fit_end: df=df[df.index<date_fit_end]
    x, y = np.log(df.DAY_GEN.values), np.log(df[yaxis])
    X = x.reshape(-1, 1)
    
    # Quantile regression fits to a specific percentile
    qr = QuantileRegressor(quantile=quantile, alpha=0.01)
    qr.fit(X, y)
    print(f'Quantile regression index = {qr.coef_[0]:.2f}')
    
    # Calculate residuals and identify inliers (points below/at the support line)
    y_pred = qr.predict(X)
    residuals = y - y_pred
    outlier_mask = residuals > np.percentile(residuals, quantile*100)
    inlier_mask = ~outlier_mask
    
    # Re-iterate linear fit on inliers to find support
    x_inliers = x[inlier_mask]
    y_inliers = y[inlier_mask]
    par = lm.Parameters()
    par.add('intercept', value=qr.intercept_)
    par.add('slope', value=qr.coef_[0], vary=1)
    gmodel = lm.models.LinearModel()
    result_support = gmodel.fit(y_inliers, par, x=x_inliers)
    print(f'Inliers only: index = {result_support.best_values["slope"]:.2f}')

    # Final linear fit using all the points, with fixed index from the support
    par.add('slope', value=qr.coef_[0], vary=0)
    result_all = gmodel.fit(y, par, x=x)
    
    if plot:
        x_plot = np.linspace(x.min(), x.max(), 100)
        fig, ax = plt.subplots(figsize=(8,6))
        ax.plot(x[inlier_mask], y[inlier_mask], 'g.', label=f'Support points (inliers, {quantile*100:.0f}%)', markersize=4)
        ax.plot(x[outlier_mask], y[outlier_mask], '.', color='grey', alpha=.4, label='Excluded (outliers)', markersize=4)

        # ax.plot(x, y_pred, 'b-', linewidth=2, label=f'Quantile fit (q={quantile})')
        ax.plot(x_plot, result_support.eval(x=x_plot), 'r--', label='Support fit')
        ax.plot(x_plot, result_all.eval(x=x_plot), 'b--', label='All dates (index from support)')
        ax.set_xlabel('ln(Days since genesis)');ax.set_ylabel(f'ln({yaxis})')
        ax.legend(framealpha=1)
        ax.grid(alpha=.5)
        plt.show()
    
    return qr, result_support, result_all

def extrapolate_PL(yaxis, best_PL, support_PL, btc_price, day_extra, date_fit_start, date_fit_end, m2_growth_rate, btc_halving_daygen, 
                   halving_mean_duration, loglogplot=0, plot_price=True, N_plot=500):
    '''extrapolate the simple PL models (support and entire dataset)
    plots the result with real dates and prices
    '''
    xplot = np.linspace(date_to_daygen(date_fit_start, genesis_block_date), date_to_daygen(date_fit_end, genesis_block_date) + day_extra, N_plot)
    dateplot = genesis_block_date + np.timedelta64(1,'D') * xplot # for the semi-log plot
    # To do: combine support/best
    df_best = pd.DataFrame({'Date':dateplot, 'DAY_GEN':xplot, yaxis:best_PL.eval(xplot)})
    df_support = pd.DataFrame({'Date':dateplot, 'DAY_GEN':xplot, yaxis:support_PL.eval(xplot)})
    if yaxis == 'Price_GEN': # convert back to real USD
        df_best['Price'] = df_best.apply(get_real_price, args=[m2_growth_rate], axis=1)
        df_support['Price'] = df_support.apply(get_real_price, args=[m2_growth_rate], axis=1)
        
    predicted_halving_daygen =  np.array([(k+len(btc_halving_daygen)+1)*halving_mean_duration for k in range(int(day_extra//halving_mean_duration)-1)], dtype='int64')
    btc_all_halving = np.concatenate([btc_halving_daygen, predicted_halving_daygen])

    if plot_price:
        fig, ax=plt.subplots(figsize=(11,6))
        if loglogplot:
            ax.plot(btc_price.DAY_GEN, btc_price['Price'], 'k', label='BTC Price')
            ax.plot(xplot, 'Price', 'blue', label='Power-law index = {:.2f})'.format(support_PL.exponent), data=df_best)
            ax.plot(xplot, 'Price', 'red', label='Support', data=df_support)
            # ax.plot(xplot, result_peak.eval(x=xplot),'r',alpha=.6, label='Top cycle (decay = {:.2f} y)'.format(result_peak.best_values['tau']/365))
            for halving_date in btc_halving_daygen: ax.axvline(halving_date,color='grey',linestyle='--',alpha=.7)
            for halving_date in predicted_halving_daygen: ax.axvline(halving_date ,color='grey',linestyle='--',alpha=.4) # 'Predicted halving'
            ax.set_xscale('log')
        else:
            ax.plot(btc_price.index, btc_price['Price'], 'k', label='BTC Price')
            ax.plot(dateplot, 'Price', 'blue', label='Power-law index = {:.2f}'.format(support_PL.exponent), data=df_best)
            ax.plot(dateplot, 'Price', 'red', label='Support', data=df_support)
            # ax.plot(dateplot, result_all.eval(x=xplot),'blue', label='Best fit (index={:.2f})'.format(result_all.best_values['exponent']))
            # ax.plot(dateplot, result_peak.eval(x=xplot),'r',alpha=.6, label='Top cycle (decay = {:.2f} y)'.format(result_peak.best_values['tau']/365))
            ax.axvspan(date_fit_start, date_fit_end, color='green', alpha=.15, label='Fitting period')
            for halving_date in btc_halving_dates: ax.axvline(np.datetime64(halving_date),color='grey',linestyle='--',alpha=.6)
            for halving_date in predicted_halving_daygen: ax.axvline(np.datetime64(halving_date+genesis_block_date) ,color='grey',linestyle='--',alpha=.4) # 'Predicted halving'

        ax.set_yscale('log')
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_log_price))
        ax.set_xlabel('Date')#;ax.set_ylabel('Price (USD)')
        ax.legend();ax.grid(alpha=.5)

    return df_best, df_support, btc_all_halving
    

################ Cycle ratio (bullrun) fitting ################

def find_cycle_k(x, btc_halving_daygen):
    '''find number of halving cycles that passed before date'''
    halving_diff = (x.DAY_GEN - btc_halving_daygen) # distance of date with each halving start
    valid_idx = np.where(halving_diff >= 0)[0] # select only positive distance
    if len(valid_idx)==0: return -1
    else: return valid_idx[halving_diff[valid_idx].argmin()] # find index of smallest positive difference

def decay_lorentz(x, A0, mu, sigma_c, d, PL_offset):
    ''' lorentzian with amplitude that decays exponentially through each k cycle + constant 
    x[0]=t: time in days, x[1]=k: cycle index
    A0: initial cycle amplitude, mu: decay ratio
    sigma_c: Lorentzian width = duration of bullrun, d: Lorentzian center = time from halving to peak in days
    PL_offset: offset from the base model (powerlaw)
    '''
    return ((A0 * mu**x[1])/np.pi) * (sigma_c / ((x[0] - d)**2 + sigma_c**2) ) + PL_offset

def decay_gauss(x, A0, mu, sigma_c, d, PL_offset):
    ''' gaussian with amplitude that decays exponentially through each k cycle + constant 
    x[0]=t: time in days, x[1]=k: cycle index
    A0: initial cycle amplitude, mu: decay ratio
    sigma_c: Gaussian width = duration of bullrun, d: center = time from halving to peak in days
    PL_offset: offset from the base model (powerlaw)
    '''
    return ((A0 * mu**x[1])/(np.sqrt(2 * np.pi)* sigma_c)) * np.exp(-((x[0] - d)**2 /(2 * sigma_c**2)) ) + PL_offset

def fit_cycle(df, yaxis, support_PL, btc_halving_daygen, start_cycle_fit=None, end_cycle_fit=None, rolling_days=10, plot_ratio_fit=True):
    # cycle ratio is defined as c = price/support - 1, where support is a PL
    # remove pre-first-halving data, smooth out details
    df['cycle_ratio'] = df[yaxis].values / support_PL.eval(df.DAY_GEN.values) - 1
    btc_cycle_fit = df.loc[(df.index>start_cycle_fit)&(df.index<end_cycle_fit)].rolling(rolling_days, center=True).mean()
    btc_cycle_fit = btc_cycle_fit.dropna(subset=['cycle_ratio']) # remove resulting nans
    btc_cycle_fit['k_cycle'] = btc_cycle_fit.apply(find_cycle_k, axis=1, args=[btc_halving_daygen])
    btc_cycle_fit['DAY_CYCLE'] = btc_cycle_fit.DAY_GEN - btc_halving_daygen[btc_cycle_fit.k_cycle] # number of days after latest halving (t')

    par_cycle= lm.Parameters()
    par_cycle.add('A0', value=10, min=0)
    par_cycle.add('mu', value=.5, min=0)
    par_cycle.add('sigma_c', value=300, min=0)
    par_cycle.add('d', value=300, min=0)
    par_cycle.add('PL_offset', value=0)
    gmodel = lm.Model(decay_lorentz)
    # gmodel = lm.Model(decay_gauss)

    result_cycle = gmodel.fit(btc_cycle_fit.cycle_ratio, par_cycle, x=[btc_cycle_fit.DAY_CYCLE, btc_cycle_fit.k_cycle] )
    # display(result_cycle)

    if plot_ratio_fit:
        fig, ax=plt.subplots(figsize=(11,6))
        ax.plot(df.index, 'cycle_ratio',color='grey',alpha=.6, data=df, label=None)
        ax.plot(btc_cycle_fit.index, 'cycle_ratio','k', label='Smoothed cycle ratio (P/S - 1)', data=btc_cycle_fit)
        ax.plot(btc_cycle_fit.index, result_cycle.eval(x=[btc_cycle_fit.DAY_CYCLE, btc_cycle_fit.k_cycle]), 'r', label='Model')
        ax.set_xlabel('Date'); ax.set_ylabel('Support ratio')
        for halving_date in btc_halving_dates:
            ax.axvline(np.datetime64(halving_date),color='grey',linestyle='--',alpha=.6)
        ax.legend()
    return result_cycle

def extrapolate_cycle(yaxis, result_cycle, support_PL, btc_price, df_best, df_support, start_cycle_fit, end_cycle_fit, btc_all_halving, m2_growth_rate, plot_cycle_fit=True):
    df_best['k_cycle'] = df_best.apply(find_cycle_k, axis=1, args=[btc_all_halving])
    df_best.drop(df_best[df_best['k_cycle'] == -1].index, inplace=True) # remove dates before 1st halving
    df_best['DAY_CYCLE'] = df_best.DAY_GEN - btc_all_halving[df_best.k_cycle] # number of days after latest halving (t')
    if yaxis == 'Price_GEN': # convert back to real USD
        df_best['Price_GEN'] = (result_cycle.eval(x=[df_best.DAY_CYCLE, df_best.k_cycle]) + 1) * support_PL.eval(df_best.DAY_GEN.values)
        df_best['Price_cycle'] = df_best.apply(get_real_price, args=[m2_growth_rate], axis=1)
    else:
        df_best['Price_cycle'] = (result_cycle.eval(x=[df_best.DAY_CYCLE, df_best.k_cycle]) + 1) * support_PL.eval(df_best.DAY_GEN.values)

    if plot_cycle_fit:
        fig, ax=plt.subplots(figsize=(11,6))
        ax.plot(btc_price.index, btc_price['Price'], 'k', alpha=.8, label='Data')
        ax.plot('Date', 'Price_cycle' ,'blue', alpha=.7, linewidth=2.5, label='Cycle fit', data=df_best)
        ax.plot('Date', 'Price', 'r--', alpha=.8, label='Support', data=df_support)
        for halving_date in btc_halving_dates: ax.axvline(np.datetime64(halving_date),color='grey',linestyle='--',alpha=.6)
        ax.axvspan(start_cycle_fit, end_cycle_fit, color='green', alpha=.15, label='Fitting region')
        ax.set_yscale('log')
        ax.set_xlabel('Date')#;ax.set_ylabel('Price (USD)')
        ax.legend();ax.grid(alpha=.5)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_log_price))
        ax.format_coord = format_coord
    return df_best


################ Plotting fct ################

def plot_rolling_avg(df, rolling_days):
    btc_price_avg = df[['Price','DAY_GEN']].rolling(rolling_days, center=False).mean()
    btc_price_avg.dropna(inplace=True)
    fig, ax=plt.subplots(figsize=(8,5))
    ax.plot('DAY_GEN', 'Price', data=btc_price_avg, label=f'Rolling average ({rolling_days} days)')
    ax.set_xlabel('Day since genesis block (2009-01-03)')
    ax.loglog()
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_log_price))
    ax.legend()
    ax.grid()
    return btc_price_avg

################ Other ################

def test_return(df, period_days):
    '''test on previous data the time scale to have positive return'''
    price_diff = df.diff(periods=period_days).Price
    is_always_positive = (price_diff.dropna() < 0).sum() == 0
    if is_always_positive: print(f'Returns are always positive for {period_days} days period')
    else: print(f'Returns not always positive for {period_days} days period')
    return is_always_positive

def find_support_current_price(df, support_PL):
    '''find the date where the support is equal current price
    this indicate the maximal date at which to expect positive returns
    '''
    curr_price, curr_date = df.iloc[-1].Price_GEN, df.iloc[-1].name
    curr_price_support_gen = (curr_price/support_PL.amplitude)**(1/support_PL.exponent)
    curr_price_support_date = genesis_block_date + np.timedelta64(1,'D') * curr_price_support_gen
    print(f'Current price of $ {curr_price:.0f} is the support of {curr_price_support_date}, in {curr_price_support_date-df.index[-1]:}')

