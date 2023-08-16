import numpy as np
from typing import List, Tuple, Dict

def covariance(
    time_serie: np.array,
    lags: List[int],
    bias: bool=False 
) -> np.array:

    """
    Computes the autocovariance for a given lag.
    Arguments:
        time_serie: A one-dimensional numpy array containing the time serie.
        lags: A list containing multiple lags to compute the autocovariace for.
        bias: If the autocovariance computing would be biased or unbiased
    Returns:
        A list containing the sampled autocovariances, one per lag in lags list.
    """

    mean = np.mean(time_serie)
    autocovariance = 0
    autocovariances = []
    for j in range(len(lags)):
        for i in np.arange(0, len(time_serie)-lags[j]):
            autocovariance += (time_serie[i]-mean)*(time_serie[i+lags[j]]-mean)
        if bias:
            autocovariances.append((autocovariance/(len(time_serie)))) 
        else:
            autocovariances.append((autocovariance/(len(time_serie)-1))) 
        autocovariance = 0
    return  autocovariances

def innovations(
    time_serie: np.array,
    order: int
    ) -> Dict[str, np.array]:

    """
    Computes the parameters of an MA(q) process.
    Arguments:
        time_serie: A one-dimensional numpy array containing the time serie.
        order: The number of time-steps considered (q) to predict the next step
    Returns:
        A dictionary containing:
            'weights': The estimated parameters. It must be two-dimensional (q, q) numpy array.
            'variances': The estimated variances. It must be one-dimensional q numpy array.
    """
    theta = np.zeros((order, order))
    ac = covariance(time_serie, np.arange(order+1))
    x_predict = np.zeros(order+1)
    theta_sum = 0
    P = np.zeros(order+1)
    P[0] = ac[0]
    P_sum = 0

    for i in range(order):
        for j in range(i+1):
            for k in range(0,j):
                theta_sum += theta[j-1][j-k-1]*theta[i][i-k]*P[k]
            theta[i][i-j] = (ac[i-j+1] - theta_sum)/P[j]
            theta_sum = 0
            x_predict[i] += theta[i][j]*(time_serie[i+1-j] - x_predict[i+1-j])

            P_sum += (theta[i][i-j]**2)*P[j]
        P[i+1] = ac[0] - P_sum
        P_sum = 0
        
       
    P = P[1:order+1]
    return {
        "weights": theta,
        "variances": P
    }

def durbin_levinson(
    time_serie: np.array,
    order: int,
    ) -> Dict[str, np.array]:

    """
    Computes the parameters of an AR(p) process.
    Arguments:
        time_serie: A one-dimensional numpy array containing the time serie.
        order: The number of time-steps considered (p) to predict the next step
    Returns:
        A dictionary containing:
            'weights': The estimated parameters. It must be two-dimensional (p, p) numpy array.
            'variances': The estimated variances. It must be one-dimensional p numpy array.
    """

    phi = np.zeros((order, order))
    P = np.zeros(order)
    ac = covariance(time_serie, np.arange(order+1))
    P[0] = ac[0]
    phi[0][0] = 0
    top_part = 0
    bot_part= 0
    for i in range(order):
        for j in range(i,-1,-1):
            if i==j:
                if i==0:
                    phi[i][j] = (ac[i+1]/ac[0])
                else:
                    for k in range(1,i+1):
                        top_part += (ac[i-k+1]/ac[0]) * phi[i-1][k-1]
                        bot_part += (ac[k]/ac[0]) * phi[i-1][k-1]
                    phi[i][j] = ((ac[i+1]/ac[0]) - top_part)/(1-bot_part)
                    top_part=0
                    bot_part=0
                    
            else:
                phi[i][j] = phi[i-1][j] - (phi[i][i]*phi[i-1][i-j-1])
        if i==0: 
            P[i] = P[0]*(1-phi[i][i]**2)
        else:
            P[i] = P[i-1]*(1-phi[i][i]**2)
        



    return {
        "weights": phi,
        "variances": P
    }

class ARMA:
    def __init__(
        self,
        p: int,
        q: int
    ):
        """
        Save the necessary information 
        Arguments:
            p: The AR process order.
            q: The MA process order.
        Attributes:
            p: The AR process order.
            q: The MA process order.
            phis: The AR process parameters.  This must be a one-dimensional numpy array of size p.
            thetas: The MA process parameters. This must be a one-dimensional numpy array of size q.
            psis: The intermediate parameters required to compute the MA and AR parameters. 
                    This must be a one-dimensional numpy array of size p+q.
            ar_roots: The roots of the associated polynomial of the AR process.  
                    This must be a one-dimensional numpy array of size p.
            ma_roots: The roots of the associated polynomial of the MA process.  
                    This must be a one-dimensional numpy array of size q.
        """
        
        self.p = p
        self.q = q
        self.phis = np.zeros(self.p)
        self.thetas = np.zeros(self.q)
        self.psis = np.zeros(self.p + self.q)

        self.ar_roots = np.zeros(self.p)
        self.ma_roots = np.zeros(self.q)

    def train(
        self,
        time_serie: np.array
    ):

        """
        Estimates the parameters psis, phis and thetas, and compute the roots of the
        associated AR and MA polynomials.

        Arguments:
            time_serie: A one-dimensional numpy array containing the time serie.
        """
        inno = innovations(time_serie, self.q+self.p)
        psis = inno["weights"][-1]
        durbin = durbin_levinson(psis, self.p)
        phis = durbin["weights"][-1]
        self.phis = phis
        self.psis = psis
        psis = np.insert(psis, 0, 1, axis=0)
        
        for i in range(self.q):
            self.thetas[i] = psis[i+1] - sum(phis[k] * psis[i-k] for k in range(i+1))
        root_ar = np.append(1, np.negative(phis))
        self.ar_roots = np.roots(root_ar[::-1])
        root_ma = np.append(1, self.thetas)
        self.ma_roots = np.roots(root_ma[::-1])
        pass
            
    def is_causal(self) -> bool:        
        """
        Indicates if the current trained ARMA process is causal.
        Returns:
            True if it is causal, False otherwise.
        """

        

        return all(np.abs(self.ar_roots) > 1 ) 

    def is_invertible(self) -> bool:
        """
        Indicates if the current trained ARMA process is invertible.
        Returns:
            True if it is invertible, False otherwise.
        """

        # CODE HERE

        return all(np.abs(self.ma_roots) > 1 )

    def forecast(
        self, 
        time_serie: np.array, 
        future_values: int = 1
        ) -> np.array:
        """
        Predicts future values with the estimated parameters.
        Arguments:
            time_serie: A one-dimensional numpy array containing the time serie.
            future_values: The number of future values to predict.
        Returns:
            The predicted future values.
        """
        
        truncated_ts = np.flip(time_serie[-(self.p + self.q + 1):])

        for t in range(1, future_values + 1):
            forecast = (truncated_ts[:-t] * self.psis).sum()
            truncated_ts = np.concatenate(([forecast], truncated_ts))

        return np.flip(truncated_ts[:future_values])