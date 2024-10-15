# Imports
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from numba import jit
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import statsmodels.api as sm


def forecastvariance(fitted_model, h):
    """
    Calculate forecast variance for a given fitted exponential smoothing model.

    Parameters:
    fitted_model : object
        A fitted Holt-Winters exponential smoothing model from `statsmodels`.
    h : int
        Forecast horizon (number of periods ahead to calculate variance for).
    
    Returns:
    result : array
        An array of forecast variances for the given horizon.
    """
    params = fitted_model.params

    # Get smoothing parameters and check for NaN, if NaN, set default
    alpha = params.get('smoothing_level', 1)
    alpha = 1 if np.isnan(alpha) else alpha  # Check for NaN and set to default

    beta = params.get('smoothing_trend', 0)
    beta = 0 if np.isnan(beta) else beta  # Check for NaN and set to default

    delta = params.get('smoothing_seasonal', 0)
    delta = 0 if np.isnan(delta) else delta  # Check for NaN and set to default

    phi = params.get('damping_trend', 1)
    phi = 1 if np.isnan(phi) else phi  # Check for NaN and set to default

    # Residual variance (sigma^2)
    sigma2 = np.mean(fitted_model.resid**2)
    var = sigma2

    # Seasonal periods
    m = fitted_model.model.seasonal_periods

    # Check if seasonal and trend components are multiplicative
    is_seasonal_multiplicative = fitted_model.model.seasonal in ["mul", "multiplicative"]
    is_trend_multiplicative = fitted_model.model.trend in ["mul", "multiplicative"]

    if not is_trend_multiplicative:
        if is_seasonal_multiplicative:
            assert h <= m, 'Forecast variance not available for h > m in the multiplicative model.'

        result = np.zeros(h)
        aux = 1  # Auxiliary variable to accumulate

        for i in range(h):
            # Initialize variance with the base residual variance (sigma^2)
            result[i] = var

            # Update aux with smoothing level, trend, and damping components
            aux += (phi ** (i + 1)) * beta
            
            if i > 0 and m> 0 and i % m == 0:
                var += (alpha * aux + delta * (1 - alpha)) ** 2 * sigma2
            else:
                var += np.power(alpha*aux,2)*sigma2

        return result
    else:
        raise ValueError('The current setting (non-multiplicative trend) is not supported.')
        

        

def intervalforecast(fitted_model, h, level=0.95):
    """
    Generate interval forecasts for an exponential smoothing model.
    
    Parameters:
    fitted_model : object
        A fitted Holt-Winters exponential smoothing model from `statsmodels`.
    h : int
        Forecast horizon (number of periods ahead to forecast).
    level : float, optional
        Confidence level for the interval (default is 0.95 for 95% confidence).
        
    Returns:
    np.array : 2D array
        Interval forecasts with lower and upper bounds.
    """
    # Calculate the residual variance (sigma^2)
    sigma2 = np.mean(fitted_model.resid**2)
    
    # Critical value for the desired confidence level
    crit = stats.norm.ppf(1 - (1 - level) / 2)
    
    # Generate forecast for the next h periods
    forecast = np.reshape(fitted_model.forecast(h).values, (h, 1))
    
    # Calculate variance for the forecast horizon
    var = np.reshape(forecastvariance(fitted_model, h), (h, 1))
    
    # Generate the lower and upper bounds for the interval
    lower_bound = forecast - crit * np.sqrt(var)
    upper_bound = forecast + crit * np.sqrt(var)
    
    # Return the forecast with lower and upper bounds
    return np.hstack((lower_bound, upper_bound))

    
    
def histogram(series):
    fig, ax= plt.subplots(figsize=(8,5))
    sns.distplot(series, ax=ax, hist_kws={'alpha': 0.8, 'edgecolor':'black'},  
                 kde_kws={'color': 'black', 'alpha': 0.7})
    sns.despine()
    return fig, ax


def qq_plot(residuals):
    fig, ax = plt.subplots(figsize=(8,5))
    pp = sm.ProbPlot(residuals, fit=True)
    qq = pp.qqplot(color='#1F77B4', alpha=0.8, ax=ax)
    a=ax.get_xlim()[0]
    b=ax.get_xlim()[1]
    ax.plot([a,b],[a,b], color='black', alpha=0.6)
    ax.set_xlim(a,b)
    ax.set_title('Normal Q-Q plot for the residuals', fontsize=12)
    return fig, ax

def plot_components_x13(results, label=''):
    colours=['#D62728', '#FF7F0E', '#2CA02C', '#1F77B4']
    fig, ax = plt.subplots(2,2, figsize=(12,8))
    ax[0,0].plot(results.observed, color=colours[0], alpha=0.95)
    ax[0,0].set(ylabel=label, title='Observed')
    ax[0,1].plot(results.trend, color=colours[1], alpha=0.95)
    ax[0,1].set(title='Trend')
    ax[1,0].plot(results.observed/results.seasadj, color=colours[2],  alpha=0.95)
    ax[1,0].set(ylabel=label, title='Seasonal')
    ax[1,1].plot(results.irregular, color=colours[3],  alpha=0.95)
    ax[1,1].set(title='Irregular')
    fig.suptitle('Time series decomposition  (X-13 ARIMA-SEATS)', fontsize=13.5)   
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    return fig, ax

def fanchart(y, forecast, intv1, intv2, intv3):
    assert isinstance(y, pd.core.series.Series), 'The time series must be a pandas series'
    assert isinstance(forecast, pd.core.series.Series), 'The forecast must be a pandas series'

    last = y.iloc[-1:]  # Extract the last value from the Series object

    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(figsize=(8, 5))
        y.plot(color='#D62728')
        extended = pd.concat([last, forecast])
        extended.plot(color='black', alpha=0.4, label='Point forecast')
        
        ax.fill_between(extended.index, pd.concat([last, intv3.iloc[:,0]]), pd.concat([last, intv3.iloc[:,1]]), facecolor='#FAB8A4', lw=0)
        ax.fill_between(extended.index, pd.concat([last, intv2.iloc[:,0]]), pd.concat([last, intv2.iloc[:,1]]), facecolor='#F58671', lw=0)
        ax.fill_between(extended.index, pd.concat([last, intv1.iloc[:,0]]), pd.concat([last, intv1.iloc[:,1]]), facecolor='#F15749', lw=0)
        hold = ax.get_ylim()
        ax.fill_betweenx(ax.get_ylim(), extended.index[0], extended.index[-1], facecolor='grey', alpha=0.15)
        ax.set_ylim(hold)
    return fig, ax

    
def sarimaforecast(y, model, h=1, m=12):
    
    n=len(y)
    x=np.zeros((n+h))
    x[:n]=y

    forecast_diff=model.forecast(steps=h)[0]

    for i in range(h):
        x[n+i]=x[n+i-1]+x[n+i-m]-x[n+i-m-1]+forecast_diff[i]
    
    return x[-h:]
    


    
    
