'''
Created on 23-Feb-2018

@author: Vishnu
'''

import numpy as np
from sklearn.externals import joblib

def calculate_score(log_odds):
    return 300 + (40 / np.log(2)) * (-log_odds)

def newscore(data):
    req_data = np.fromstring(data, dtype=int, sep=',')
#     model = joblib.load('/home/dev/FraudDetection/FraudDetection/creditscorenew.pkl')
    model = joblib.load('C:/Users/hp/eclipse-workspace/FraudDetection/FraudDetection/creditscorenew.pkl')
    log_prob = model.predict_log_proba([req_data])[:,1]
    score = calculate_score(log_prob)
    return score[0]