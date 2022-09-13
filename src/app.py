import numpy as np
from flask import Flask, request,render_template
from model import model_predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    l = [x for x in request.form.values()]
    company = l[0]
    year = int(l[3])
    quarter = int(l[2])
    location = l[1]

    prediction = model_predict(company, year, quarter, location)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Quarter profits should be $USD {output}')

if __name__ == "__main__":
    app.run(debug=True)