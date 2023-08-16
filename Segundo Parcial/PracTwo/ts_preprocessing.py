import numpy as np
from typing import Dict
from sklearn.linear_model import LinearRegression


def autocovariance (
        time_serie: np.array,
        lag: int=1
    ) -> float:

    """
    Computes the unbias autocovariance for a given lag.
    Arguments:
        time_serie: A one-dimensional numpy array containing the time serie.
        lag: An integer indicating the lag autocovariance computing for.
    Returns:
        The unbias sampled autocovariance.
    """

    mean = np.mean(time_serie)
    autocovariance = 0

    for i in np.arange(0, len(time_serie)-lag):
        autocovariance += (time_serie[i]-mean)*(time_serie[i+lag]-mean)
    return (1/(len(time_serie)))*autocovariance


def detrending (
        time_serie: np.array
    ) -> Dict[str, float]:

    """
    Computes a linear regression respect to time to approximate the trend of a time serie.
    Arguments:
        time_serie: A one-dimensional numpy array containing the time serie.
    Returns:
        The bias and weight of the fitted model.
    """
    x = np.arange(0, len(time_serie))
    x = x.reshape(-1,1)

    model = LinearRegression()
    model.fit(x, time_serie)

    return {
        "bias": model.intercept_,
        "weight": model.coef_[0]
    }


def differencing (
    time_series: np.array
    ) -> np.array:

    """
    Removes the trending of a time serie by differencing method
    Arguments:
        time_serie: A one-dimensional numpy array containing the time serie.
    Returns:
        The detrended time serie.
    """

    return (np.concatenate((time_series[1], ([time_series[i+1]-time_series[i] for i in range(len(time_series)-1)])), axis=None))