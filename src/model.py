# Importing the libraries
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.linear_model import LinearRegression
import os
from datetime import date 

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
                train([X], [y], MODEL_PATH)


def predict(company, year, month, day, city):

    date = date(year, month, day)

    quarter = (date.month - 1) // 3 + 1

    path = os.path.join('models', company, str(quarter), city)
    with open(path, 'rb') as f:
        model = pickle.load(f)

    return model.predict([date.year])

if __name__ == '__main__':
    generate_models()
    #print(predict('Pink Cab', 2022, 4, 8, 'NEW YORK NY'))

        

    # Loading model to compare the results

    # print(model.predict([[2, 2200, 5]]))