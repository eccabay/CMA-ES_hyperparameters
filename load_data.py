from sklearn import datasets
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# Multiclass Classification Datasets

def load_mnist():
    mnist = datasets.load_digits(as_frame=True)
    mnist_X, mnist_y = mnist.data, mnist.target
    
    X_train, X_test, y_train, y_test = train_test_split(mnist_X, mnist_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_forest_covertypes():
    forest = datasets.fetch_covtype(as_frame=True)
    forest_X, forest_y = forest.data, forest.target
    forest_X = forest_X[:15000]
    forest_y = forest_y[:15000]
    
    X_train, X_test, y_train, y_test = train_test_split(forest_X, forest_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_kepler_exoplanets():
    kepler = pd.read_csv('data/kepler_exoplanet.csv')
    not_X_columns = ['rowid', 'kepid', 'kepoi_name', 'kepler_name', 'koi_score', 'koi_pdisposition', 'koi_disposition', 'koi_teq_err1', 'koi_teq_err2', 'koi_tce_delivname']

    kepler_X = kepler.drop(not_X_columns, axis=1)
    kepler_y = kepler['koi_pdisposition']
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    kepler_X_array = imputer.fit_transform(kepler_X)
    kepler_X = pd.DataFrame(kepler_X_array, columns=kepler_X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(kepler_X, kepler_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test 


# Regression Datasets

def load_cali_housing():
    california = datasets.fetch_california_housing(as_frame=True)
    cali_X, cali_y = california.data, california.target
    
    X_train, X_test, y_train, y_test = train_test_split(cali_X, cali_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test    

def load_melbourne_housing():
    melbourne = pd.read_csv('data/melbourne_housing_data.csv')

    not_X_columns = ['Address', 'Price', 'SellerG', 'Date']
    melbourne_X = melbourne.drop(not_X_columns, axis=1)
    melbourne_y = melbourne['Price']

    # Convert all features to integers
    categorical_cols = ['Suburb', 'Type', 'Method', 'CouncilArea', 'Regionname']
    for col_name in categorical_cols:
        melbourne_X[col_name] = pd.Categorical(melbourne_X[col_name]).codes

    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    melbourne_X_array = imputer.fit_transform(melbourne_X)
    melbourne_X = pd.DataFrame(melbourne_X_array, columns=melbourne_X.columns)

    X_train, X_test, y_train, y_test = train_test_split(melbourne_X, melbourne_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_world_happiness():
    happiness = pd.read_csv('data/world-happiness-report-2021.csv')

    X_columns = ['Regional indicator', 'Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']
    happiness_X = happiness[X_columns]
    happiness_y = happiness['Ladder score']

    happiness_X['Regional indicator'] = pd.Categorical(happiness_X['Regional indicator']).codes

    X_train, X_test, y_train, y_test = train_test_split(happiness_X, happiness_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# Binary Classification Datasets

def load_heart_attack():
    heart_attack = pd.read_csv('data/heart-attack.csv')

    heart_X = heart_attack.drop('output', axis=1)
    heart_y = heart_attack['output']

    X_train, X_test, y_train, y_test = train_test_split(heart_X, heart_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_stroke():
    stroke = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

    not_X_columns = ['id', 'stroke']
    stroke_X = stroke.drop(not_X_columns, axis=1)
    stroke_y = stroke['stroke']

    # Convert all features to integers
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col_name in categorical_cols:
        stroke_X[col_name] = pd.Categorical(stroke_X[col_name]).codes

    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    stroke_X_array = imputer.fit_transform(stroke_X)
    stroke_X = pd.DataFrame(stroke_X_array, columns=stroke_X.columns)

    X_train, X_test, y_train, y_test = train_test_split(stroke_X, stroke_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def load_telecom():
    telecom = pd.read_csv('data/telecom_users.csv')

    not_X_columns = ['Unnamed: 0', 'customerID', 'TotalCharges', 'Churn']
    telecom_X = telecom.drop(not_X_columns, axis=1)
    telecom_y = pd.Categorical(telecom['Churn']).codes

    # Convert all features to integers
    not_categorical_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges']
    for col_name in telecom_X.columns:
        if col_name not in not_categorical_cols:
            telecom_X[col_name] = pd.Categorical(telecom_X[col_name]).codes

    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    telecom_X_array = imputer.fit_transform(telecom_X)
    telecom_X = pd.DataFrame(telecom_X_array, columns=telecom_X.columns)

    X_train, X_test, y_train, y_test = train_test_split(telecom_X, telecom_y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def load_multiclass():
    all_data = {}
    
    all_data['mnist'] = load_mnist()
    all_data['forest_covertypes'] = load_forest_covertypes()
    all_data['kepler_exoplanets'] = load_kepler_exoplanets()
    
    return all_data

def load_regression():
    all_data = {}
    
    all_data['california_housing'] = load_cali_housing()
    all_data['melbourne_housing'] = load_melbourne_housing()
    all_data['world_happiness'] = load_world_happiness()
    
    return all_data

def load_binary():
    all_data = {}
    
    all_data['heart_attack'] = load_heart_attack()
    all_data['stroke'] = load_stroke()
    all_data['telecom'] = load_telecom()
    
    return all_data
