'''
Created on 22-Feb-2018

@author: Vishnu
'''

import numpy as np
from sklearn.externals import joblib
import pandas as pd

def calculate_score(log_odds):
    return 300 + (40 / np.log(2)) * (-log_odds)

def is_poor_coverage(row):
    pct_null = float(row.isnull().sum()) / row.count()
    return pct_null < 0.8

def score(data):
    data = data.split(',')
    df = pd.DataFrame(columns=["issue_d", "loan_status", "home_ownership", "last_fico_range_low", "last_fico_range_high"], data=[data])
    df = df[df.apply(is_poor_coverage, axis=1)]
    df['year_issued'] = df.issue_d.apply(lambda x: int(x.split("-")[0]))
    df_term = df[df.year_issued < 2025]
    bad_indicators = ["Late (16-30 days)", "Late (31-120 days)", "Default", "Charged Off"]
    df_term['is_rent'] = df_term.home_ownership=="RENT"
    df_term['is_bad'] = df_term.loan_status.apply(lambda x: x in bad_indicators)
    features = ['last_fico_range_low', 'last_fico_range_high', 'is_rent']
    model = joblib.load('/home/dev/FraudDetection/FraudDetection/creditscore.pkl')
    log_probs = model.predict_log_proba(df_term[features])[:,1]
    score = calculate_score(log_probs)
    return score[0]

    