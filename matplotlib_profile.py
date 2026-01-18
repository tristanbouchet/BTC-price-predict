import matplotlib.dates as mdates
import matplotlib as mpl

mpl.rcParams['axes.labelsize'] = 15
mpl.rcParams['axes.titlesize'] = 15
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 14

def fmt_log_price(y, pos):
    '''Format log-scale axis: plain numbers with no trailing zeros'''
    return f'${y:,.10g}'

def format_coord(x, y):
    '''Format cursor coordinates with full date and formatted price'''
    # x is matplotlib date number, convert to datetime
    date = mdates.num2date(x)
    formatted_date = date.strftime("%d-%m-%Y")  # Full date format
    formatted_price = f'{y:,.10g}'  # Price with commas and no trailing zeros
    return f'{formatted_date}, ${formatted_price}'
