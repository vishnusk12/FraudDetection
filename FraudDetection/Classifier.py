'''
Created on 21-Feb-2018

@author: Vishnu
'''

from sklearn.externals import joblib
import numpy as np

def model(data):
    data_list = data.split(',')
    data_list.pop(0) 
    req_data = ','.join(data_list)
    req_data = np.fromstring(req_data, dtype=int, sep=',')
    model = joblib.load('C:/Users/hp/Documents/Python Scripts/Fraud Detection/fraud_model.pkl')
    prediction = model.predict([req_data])
    return prediction[0]