# Importing the libraries
import pickle
import json
from sklearn.linear_model import LinearRegression
import os
import numpy as np

with open('regression_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

with open('queryData.json', 'r') as f:
    queryData = json.load(f)

def train(X, y, path):
    regressor = LinearRegression()
    regressor.fit(X, y)
    pickle.dump(regressor, open(os.path.join(path, 'model.pkl'),'wb'))


def generate_models():
    for company in queryData['company']:
        _company = queryData['company'][company]
        for city in queryData['city']:
            _city = queryData['city'][city]
            for quarter in queryData['quarter']:
                _quarter = queryData['quarter'][quarter]

                MODEL_PATH = os.path.join('models', company, quarter, city)
                query = f'Company == {_company} & City == {_city} & Quarter == {_quarter}'
            
                data = dataset.query(query).groupby('Year', as_index=True).sum()
                X = data.index
                y = data['Profit']
                train(np.reshape(X, (-1, 1)), y, MODEL_PATH)


def generate_overalls():
    for company in queryData['company']:
        _company = queryData['company'][company]
        for quarter in queryData['quarter']:
            _quarter = queryData['quarter'][quarter]
            MODEL_PATH = os.path.join('models', company, quarter, 'US')
            query = f'Company == {_company} & Quarter == {_quarter}'
        
            data = dataset.query(query).groupby('Year', as_index=True).sum()
            X = data.index
            y = data['Profit']
            train(np.reshape(X, (-1, 1)), y, MODEL_PATH)


def model_predict(company, year, quarter, location):

    cwd = os.getcwd()
    path = os.path.join(cwd, 'models', company, str(quarter), location, 'model.pkl')
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model.predict([[year]])


#for location in queryData['city']:
 #   for result in predict('Yellow Cab', 2022, 4, location):
  #      print(result)
