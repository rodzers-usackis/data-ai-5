# In[110]:


# import all the libraries that you need at the top of the notebook
import pandas as pd
import numpy as np
import pickle
import sklearn.utils

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor


# In[111]:


# Define BUCKET_ROOT (for now a dummy value will do, this will become clear in Part 3)
# Define DATA_DIR (initially this will be a local directory, but later it will be a Google Cloud Storage bucket)

BUCKET_ROOT = '/gcs/data_ai_5_bucket'
DATA_DIR = '/data_directory'
DATASET_FILENAME = '/kc_house_data.csv'
DATASET_LOCATION = BUCKET_ROOT + DATA_DIR + DATASET_FILENAME

MODEL_DIRECTORY = '/model_artifacts'
MODEL_FILENAME = '/model.pkl'


# In[112]:


def load_data(data_dir):
    # load the data with correct data types
    df = pd.read_csv(data_dir)

    # return the data
    return df


# In[113]:


def transform_data(df):
    # transform the data
    df.drop(columns=['id'], inplace=True)

    df['date'] = pd.to_datetime(df['date'])

    df['zipcode'] = df['zipcode'].astype(str)

    one_hot = OneHotEncoder()
    encoded = one_hot.fit_transform(df[['zipcode']])
    df[one_hot.categories_[0]] = encoded.toarray()

    df.drop('zipcode', axis=1, inplace=True)

    df.drop(['lat', 'long'], axis=1, inplace=True)

    # return the transformed data
    return df


# In[114]:


def remove_outliers(df):
    dataframe_with_removed_outliers = df.copy()

    index_names = dataframe_with_removed_outliers[dataframe_with_removed_outliers['bedrooms'] > 13].index
    dataframe_with_removed_outliers.drop(index_names, inplace=True)

    index_names = dataframe_with_removed_outliers[dataframe_with_removed_outliers['price'] > 6000000].index
    dataframe_with_removed_outliers.drop(index_names, inplace=True)

    index_names = dataframe_with_removed_outliers[dataframe_with_removed_outliers['sqft_living'] > 10000].index
    dataframe_with_removed_outliers.drop(index_names, inplace=True)

    return dataframe_with_removed_outliers


# In[115]:


def split_data(df):
    # split the data
    x = df.loc[:, ~dataframe.columns.isin(['price', 'date'])].values
    y = df['price'].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

    # return the train and test data
    return x_train, x_test, y_train, y_test


# In[116]:


def normalize_data(x_train, x_test, y_train, y_test):
    # standardize the data
    mean_x = x_train.mean()
    std_x = x_train.std()

    mean_y = y_train.mean()
    std_y = y_train.std()

    x_train_norm = (x_train - mean_x) / std_x
    x_test_norm = (x_test - mean_x) / std_x

    y_train_norm = (y_train - mean_y) / std_y
    y_test_norm = (y_test - mean_y) / std_y

    # return the standardized data
    return x_train_norm, x_test_norm, y_train_norm, y_test_norm


# In[117]:


def create_model():
    # create the ensemble model
    mlr = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()
    rfr = RandomForestRegressor()
    svr = SVR()

    ensemble = VotingRegressor(estimators=[('mlr', mlr), ('ridge', ridge), ('lasso', lasso), ('rfr', rfr), ('svr', svr)])

    # return the model
    return ensemble


# In[118]:


def train_model(model, x_train_norm, y_train_norm):
    # train the ensemble model
    model.fit(x_train_norm, y_train_norm)

    # return the model
    return model


# In[119]:


def save_model(model, root, directory, filename):
    # save the model
    filepath = root + directory + filename

    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    # return the path to the saved model
    return filepath


# In[120]:


def load_model(fp):
    # open the file
    model_file = open(fp, 'rb')

    # deserialize the model
    model = pickle.load(model_file)

    # close the file
    model_file.close()

    # return the model
    return model


# In[121]:


# apply the methods in the correct order
dataframe = load_data(DATASET_LOCATION)
transform_data(dataframe)
remove_outliers(dataframe)

x_train, x_test, y_train, y_test = split_data(dataframe)

x_train_normalized, x_test_normalized, y_train_normalized, y_test_normalized = normalize_data(x_train, x_test, y_train, y_test)

ensemble_model = create_model()
ensemble_model = train_model(ensemble_model, x_train, y_train)

filepath = save_model(ensemble_model, BUCKET_ROOT, MODEL_DIRECTORY, MODEL_FILENAME)


# In[122]:


# test if model was saved successfully
loaded_model = load_model(filepath)


# In[123]:


print(loaded_model)


# In[124]:


print(f'Ensemble model score: {loaded_model.score(x_test, y_test)}')