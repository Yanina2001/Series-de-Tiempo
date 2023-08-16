import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Dict

def interpolate (
        data: np.array, 
        window_size: int=1
    ) -> np.array:

    if window_size % 2 == 0:
        window_size = window_size + 1

    if window_size > len(data):
        window_size = len(data)

    if np.isnan(data).all():
        return data

    if np.isnan(data).any():

        for i in range(len(data)):
            if pd.isna(data[i]):
                emp_idx = max(0, i - int(window_size))
                fin_idx = min(len(data), i + int(window_size) + 1)

                for j in range(i+1, fin_idx):
                    if pd.isna(data[j]):
                        fin_idx=j
                        
                ww_data = data[emp_idx:fin_idx]
                avg_value = np.nanmean(ww_data)

                data[i] = avg_value

    return data

def get_X_y(
        time_serie: np.array,
        p: int = 1
    ) -> Dict[str, np.array]:

    """
    Formats a time serie to be as a matrix of data and the expected output after predictions.
    Arguments:
        time_serie: A one-dimensional numpy array containing the time serie.
        p: An integer indicating the previous time-steps used to predict the current value of 
            the time serie.
    Returns:
        A dictionary containing the data as a two dimensional numpy array and the expected 
        output as a one-dimensional numpy array.
    """

    data = []
    output = []
    
    for i in range(p,len(time_serie)):
        output.append(time_serie[i])
        data.append(time_serie[i-p:i])
        
    data = np.array(data)
    output = np.array(output)

    return {"data": data, "output": output}

class AutoRegressive:
    
    """
    An object to sotre the AR(p) model parameters estimations
    """

    def __init__(self, p: int):

        """
        Constructor of the object.
        Arguments:
            p: An integer indicating the previous time-steps used to predict the current value of 
                the time serie (The order of the AR(p) model). 
        """

        self.p: int = p
        self.bias: float = None
        self.weights: np.array  = None
        self.variance: float = None


    def train (
        self, 
        data: np.array, 
        output: np.array):
        
        """
        Fits an AR(p) model. The parameters are expected to be saved in the arguments of the object.
        
        Arguments:
            data: A two-dimensional numpy array containing the formated time serie as a matrix.
            output: A one-dimensional numpy array containing the expected outputs.
        """

        print(data)
        print(data.shape,output.shape)

        liR = LinearRegression()
        liR.fit(data,output)

        self.bias = liR.intercept_
        self.weights = liR.coef_

        pred = liR.predict(data)
        res = output - pred

        self.variance =  sum(res**2)/len(output-1)
        self.model = liR


    def predict (
        self, 
        data: np.array) -> np.array:

            """
            Once trained, predicts the output for a given data.
            Arguments:
                data: A two-dimensional numpy array containing the formated time serie as a matrix.
            Returns:
                A one-dimensional numpy array containing the predictions of the model.
            """

            return np.array(self.model.predict(data))