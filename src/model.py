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

    #Fitting model with trainig data
    regressor.fit(X, y)

    # Saving model to disk
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


def predict(company, year, quarter, city):

    cwd = os.getcwd()
    path = os.path.join(cwd, 'models', company, str(quarter), city, 'model.pkl')
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model.predict([[year]])


#generate_models()
for result in predict('Yellow Cab', 2022, 4, 'NEW YORK NY'):
    print(result)

        

    # Loading model to compare the results

    # print(model.predict([[2, 2200, 5]]))