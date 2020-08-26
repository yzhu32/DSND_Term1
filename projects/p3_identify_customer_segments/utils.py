# utils helper functions
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport

from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA    
from sklearn.preprocessing import LabelEncoder


def validate_int(vals):
    """
    a validation function to validate 
    integer or not
    """
    try:
        int(vals)
        return True
    except:
        return False
    

def generation_mapping(row):
    """
    Map generation
    """
    generation_dict = { 0: [1, 2], 
                        1: [3, 4], 
                        2: [5, 6], 
                        3: [7, 8, 9], 
                        4: [10, 11, 12, 13], 
                        5: [14, 15]
                      }
    try:
        for key, val in generation_dict.items():
            if row in val:
                return key
    except:
        return np.nan
    

def get_dic(null_data, df):
    """
    Get the percentage of the null values from azdias
    """
    dic = {'number of null values': null_data.values, 
           'percentage of null values': np.round(null_data.values*100/len(df), 2)}
    
    return dic


def movement_mapping(row):
    """
    Map movement
    """
    mainstream = [1, 3, 5, 8, 10, 12, 14]

    try:
        if row in mainstream:
            return 0
        else:
            return 1
    except:
        return np.nan
    

def wealth_mapping(row):
    """
    Map wealth
    """
    if pd.isnull(row):
        return np.nan
    else:
        return int(str(row)[0])


def lifestage_mapping(row):
    """
    Map life 
    """
    if pd.isnull(row):
        return np.nan
    else:
        return int(str(row)[1])

