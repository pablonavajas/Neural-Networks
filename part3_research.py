from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# External library to process categorical data
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


import pandas as pd

def process():
    l1 = pd.read_csv("part3_training_data_short.csv")

    ############################################################################################################
    #                          HIGH CARDINALITY                                                                #
    #                                                                                                          #
    # ...high_card = [col for col in l1np.select_dtypes(exclude=np.number) if l1np[col].nunique() > max_card]  #
    ############################################################################################################
    
    # Too high cardinality columns (to splitted)
    max_card = 7 #number of cases studied here

    # find non_numerical columns
    non_num = l1.select_dtypes(exclude=np.number)

    # select columns with too high cardinality
    too_long = [col for col in non_num if non_num[col].nunique() > max_card]
    
    # drop columns with excessive cardinality
    l1 = l1.drop(columns=too_long)

    
    l1np = l1.to_numpy()

    imputer = SimpleImputer(missing_values = np.nan, strategy='constant')
    imputer = imputer.fit(l1np[:,1:])
    l1np[:, 1:] = imputer.transform(l1np[:,1:])
    

    label_coder = LabelEncoder()

    for col in range(len(l1np[0])):
        if not (isinstance(l1np[0][col], (int)) or isinstance(l1np[0][col], (float))):
            l1np[:,col] = label_coder.fit_transform(l1np[:,col])

    col_trans = ColumnTransformer([('encoder', OneHotEncoder(categories='auto'), [0])],
                                          remainder='passthrough')
    X = np.array(col_trans.fit_transform(l1np), dtype = np.str)

    return X

######################

from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import seaborn as sns

def gmm_pred(X):

    sns.set()

    """
    n_components = np.arange(1,9)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(X) for n in n_components]

    plt.plot(n_components, [m.bic(X) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(X) for m in models], label='AIC')

    plt.legend(loc='best')

    plt.show()
    """
    gmm = GaussianMixture(n_components = 2)

    gmm.fit(X)

    labels = gmm.predict(X)

    print(labels) 
    
    
    return 0 #labels


if __name__=='__main__':

    x = process()

    gmm_pred(x)
