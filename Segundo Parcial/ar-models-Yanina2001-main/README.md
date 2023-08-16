# AR Models

In this practice you will implement some basics functions for the tratment of a time serie.

Here is contained a file named _ar_model.py_, where are some functions that you are required to implement in order to grade successfully your tests. You must grade successfully as much tests as possible before the deadline.

The description of each function is the same script, anyway, here are some examples of some cases that you need to cover:

## Implementations
### interpolate
Imputates the null values in a time series by the average on 'window_size' neighbors. The null values must be evaluated with the functions _numpy.isnan_ and _pandas.isna_.

**Arguments**:

- data: A one-dimensional numpy array containing the time serie.

- window_size: An integer idicating the mask size indicating how many neighbors will be used to fill the null value.

**Returns**:

A one-dimensional numpy array with the imputated time serie.

**Examples**:

_Data without null_

    data=np.array([1, 2, 3, 4, 5, 6])
    window_size=2
    output=np.array([1, 2, 3, 4, 5, 6])

_Simple case_

    data=np.array([0, 1, np.nan, 2, 3])
    window_size=2
    output=[0, 1, 6/4, 2, 3]

_Window size too large_

    data=np.array([1, 2, np.nan , 4, 5, 6])
    window_size=5
    # The mask must be truncated until the start or 
    # the end of the time serie
    output=[0, 1, 6/4, 2, 3]


_Null values in mask_

    data=np.array([0, 1, np.nan, 2, 3, np.nan, 4, 5, 6])
    window_size=5
    # First nan imputation
    # Despite the mask size is 5 it is truncated 
    # to the first found null
    # (0 + 1 + 2 + 3)/4=6/4
    np.array([0, 1, 6/4, 2, 3, np.nan, 4, 5, 6])
    # The second imputation must consider the previous
    # imputated data
    # (0 + 1 + 6/4 + 2 + 3 + 4 + 5 + 6)/8 = (21 + 6/4)/8
    output=np.array([0, 1, 6/4, 2, 3, (21 + 6/4)/8, 4, 5, 6] 

### get_X_y
Formats a time serie to be as a matrix of data and the expected output after predictions.
        
**Arguments**:

- time_serie: A one-dimensional numpy array containing the time serie.

- p: An integer indicating the previous time-steps used to predict the current value of the time serie.

**Returns**:

A dictionary containing the data as a two dimensional numpy array and the expected output as a one-dimensional numpy array.

**Examples**:

_Simple example_

    time_serie=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    p=3
    output={
        "data": np.arrray([
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9]
        ]),
        "output": np.array([3, 4, 5, 6, 7, 8, 9, 10])
    }

### Autoregressive.train
Fits an AR(p) model. The parameters are expected to be saved in the arguments of the object.        
        
**Arguments**:
- data: A two-dimensional numpy array containing the formated time serie as a matrix.

- output: A one-dimensional numpy array containing the expected outputs.

**Examples**:

_Simple case_

The bias is the $\phi_0$ parameter, weights are the $\phi_1, \cdots ,\phi_p$ parameters and the estimated variance is $\sigma_w^2$

    self.p=3
    data=np.arrray([
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9]
        ])
    output=np.array([3, 4, 5, 6, 7, 8, 9, 10])
    # Code your training phase
    self.bias=# Your computed bias 
    self.weights=# Your computed weights
    self.variance=# Your estimated variance

### Autoregressive.predict
Once trained, predicts the output for a given data.
        
**Arguments**:
- data: A two-dimensional numpy array containing the formated time serie as a matrix.

**Returns**:

A one-dimensional numpy array containing the predictions of the model.

**Examples**:

_Simple case_

    data=np.arrray([
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9]
        ])
    output=np.array([3, 4, 5, 6, 7, 8, 9, 10])

You can use the dataset _forex_mxn_usd.csv_ to test your implementations. I encourage you to code your own unittests, anyway, your functions will be tested with each commit you do on this repo.

Don't hesitate to write me if you have any doubt.

Good luck!

