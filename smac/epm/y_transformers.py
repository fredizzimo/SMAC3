import numpy as np

from sklearn.preprocessing import QuantileTransformer

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"

class AbstractYTransformer(object):
    
    def fit_transform(self, y):
        '''
            fit and transform y into target space
            
            Arguments
            ---------
            y: np.ndarray[n_samples]
            
            
            Returns
            -------
            np.ndarray[n_samples]
        '''
         
        raise NotImplemented
    
    def inverse_mean_transform(self, y):
        '''
            inverse transform y into original space
            and average across all entries in y
            
            Arguments
            ---------
            y: np.ndarray[n_samples]
            
            
            Returns
            -------
            np.ndarray[n_samples]
        '''
        
        raise NotImplemented

class IDYTransformer(AbstractYTransformer):
    
    def fit_transform(self, y):
        '''
            Returns y
            
            Arguments
            ---------
            y: np.ndarray[n_samples]
            
            
            Returns
            -------
            np.ndarray[n_samples]
        '''
        return y
    
    def inverse_mean_transform(self, y):
        '''
            Applies np.mean(y)
            
            Arguments
            ---------
            y: np.ndarray[n_samples]
            
            
            Returns
            -------
            float
        '''
        return np.mean(y)
    
class LogYTransformer(AbstractYTransformer):
    
    def fit_transform(self, y):
        '''
            Applies np.log10 to y
            
            Arguments
            ---------
            y: np.ndarray[n_samples]
            
            
            Returns
            -------
            np.ndarray[n_samples]
        '''
        return np.log10(y)
    
    def inverse_mean_transform(self, y):
        '''
            Applies np.mean(np.power(10,y))
            
            Arguments
            ---------
            y: np.ndarray[n_samples]
            
            
            Returns
            -------
            float
        '''
        return np.mean(np.power(10,y))
    
class LogNormYTransformer(AbstractYTransformer):
    
    def fit_transform(self, y):
        '''
            Applies np.log10 to y
            
            Arguments
            ---------
            y: np.ndarray[n_samples]
            
            
            Returns
            -------
            np.ndarray[n_samples]
        '''
        return np.log10(y)
    
    def inverse_mean_transform(self, y):
        '''
            Applies log normal distribution transformation
            
            Arguments
            ---------
            y: np.ndarray[n_samples]
            
            
            Returns
            -------
            float
        '''
        log_mean = np.mean(y)
        log_var = np.var(y)
        # mean of log-normal distribution:
        return 10**(log_mean + log_var/2)

class QuantileYTransformer(AbstractYTransformer):
    
    def __init__(self):
        self._transformer = QuantileTransformer(
            output_distribution='normal', 
            random_state=1234)
    
    def fit_transform(self, y):
        '''
            Applies sklearn.preprocessing.QuantileTransformer
            with output_distribution='normal'
            
            Arguments
            ---------
            y: np.ndarray[n_samples]
            
            
            Returns
            -------
            np.ndarray[n_samples]
        '''
        return self._transformer.fit_transform(y.reshape(-1, 1)).flatten()
    
    def inverse_mean_transform(self, y):
        '''
            Applies inverse transform of
            sklearn.preprocessing.QuantileTransformer 
            
            Arguments
            ---------
            y: np.ndarray[n_samples]
            
            
            Returns
            -------
            float
        '''
        return np.mean(self._transformer.inverse_transform(
                        y.reshape(-1,1)).flatten())
    