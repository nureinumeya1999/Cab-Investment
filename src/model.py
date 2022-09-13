# Importing the libraries
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.linear_model import LinearRegression
import os

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
                data = dataset.query(query)
                
                X = data.iloc[:, :-1]
                X.drop(columns=['Year', 'Quarter', 'Gender', 'Salary Group', 'Age Group'], inplace=True)
                y = data['Profit']
                train(X, y, MODEL_PATH)


def predict(input):
    company, quarter, city = input[0], input[1], input[2]
    path = os.path.join('models', company, quarter, city)
    with open(path, 'rb') as f:
        model = pickle.load(f)

    companyID, quarterID, cityID = queryData['company'][company], queryData['quarter'][quarter], queryData['city'][city]
    print(model.predict([input]))

if __name__ == '__main__':
    generate_models()

        

    # Loading model to compare the results

    # print(model.predict([[2, 2200, 5]]))