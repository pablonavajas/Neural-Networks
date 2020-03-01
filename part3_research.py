from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# External library to process categorical data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer #Encoder, OneHotEncoder
#from sklearn.compose import ColumnTransformer

import DataRead

import pandas as pd

"""class DataFrameImputer(TransformerMixin):

    def __init__(self):

    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
                               if X[c].dtype == np.dtype('0') else X[c]
"""

def process(X_raw):
    

    ############################################################
    # HIGH CARDINALITY  (not implemented in solution ... yet)  #
    #                                                          #
    ############################################################
    #...high_card = [col for col in l1np.select_dtypes(exclude=np.number) if l1np[col].nunique() > max_card]
    """
    Unsuccesful use of 1-HOT ENCODER:

    for col in strlist: #range(len(l1np[0])):
    l1np[:,col] = label_coder.fit_transform(l1np[:,col])
    #l1np[:,col] = np.reshape(label_coder.fit_transform(l1np[:,col]), (len(l1np)))
    
    #if not (isinstance(l1np[0][col], (int)) or isinstance(l1np[0][col], (float))):
    #l1np[:,col] = label_coder.fit_transform(l1np[:,col])

    col_trans = ColumnTransformer([('encoder', OneHotEncoder(categories='auto'), [0])],  
                                   remainder='passthrough')

    X = np.array(col_trans.fit_transform(l1np), dtype = np.int)
    """

    #X_raw = pd.read_csv("part3_training_data.csv")

    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import LabelBinarizer
    """
    
    # Establish a maximum cardinality for categorical variables
    max_card = 70

    # Find all categorical columns
    categorical_cols = X_raw.select_dtypes(exclude=np.number)

    # Select columns with too high cardinality
    excs_card_cols = [col for col in categorical_cols if categorical_cols[col].nunique() > max_card]

    print(excs_card_cols)
    
    # drop columns with excessive cardinality
    X_raw = X_raw.drop(columns=excs_card_cols)

    # ... continue with solution
    # listp = [i[p] for i in l1np for p in range(len(l1np[0])) if isinstance(i[p], (str))]
    
    X_np = X_raw.to_numpy()

    print(X_np[0])
    
    imputer_str = SimpleImputer(missing_values = np.nan, strategy='constant', fill_value='missing_val')
    imputer_num = SimpleImputer(missing_values = np.nan, strategy='constant', fill_value=-1)
    
    strlist = [i for i in range(len(X_np[0])) if isinstance(X_np[0][i], str) or isinstance(X_np[1][i],str)]
    
    for col in range(len(X_np[0])):
        if col in strlist:
            imputer_str = imputer_str.fit(X_np[:,col:col+1])
            X_np[:,col:col+1] = imputer_str.transform(X_np[:,col:col+1])
        else:
            imputer_num = imputer_num.fit(X_np[:,col:col+1])
            X_np[:,col:col+1] = imputer_num.transform(X_np[:,col:col+1])
    

    label_coder = LabelBinarizer()

    for col in range(len(X_np[0])):
        if col in strlist:
            onehot = label_coder.fit_transform(X_np[:,col])
            if col == 0:
                X = onehot
            else:
                X = np.concatenate((X, onehot), axis=1)
        else:
            if col == 0:
                X = X_np[:,col:col+1]
            else:
                X = np.concatenate((X, X_np[:,col:col+1]), axis=1)
        
    return X

######################

from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import seaborn as sns



if __name__=='__main__':

    dat = pd.read_csv("part3_training_data.csv")

    print(dat)
    
    datnp = dat.to_numpy()

    print(




    """
    #TODO - Need to split off a claims_raw np.array
    #Calling it claims_raw in code below
    X_raw = dat.drop(columns=["claim_amount", "made_claim"])

    y_raw = dat["made_claim"]
    y_raw = y_raw.to_numpy()
    
    claims_raw = dat["claim_amount"]
    claims_raw = claims_raw.to_numpy()

    #TODO - Split into appropriate sets (maybe training and test, then will 
    #pass training into the fit() function of the classifier and split this into
    #training and validation within the function itself for training
    #Remember to randomise this dataset before doing splits.
    dataset = DataRead.balance_and_split_into_train_valid_test(X_raw, y_raw)
    train_att, train_lab, valid_att, valid_lab, test_att, test_lab = dataset

    x = process(train_att)

    print(x[0])
    print(len(x[0]))
    
    """
