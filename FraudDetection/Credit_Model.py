'''
Created on 22-Feb-2018

@author: Vishnu
'''

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

df = pd.read_csv('C:/Users/hp/Documents/Python Scripts/Fraud Detection/LoanStats3a.csv', skiprows=1)
def is_poor_coverage(row):
    pct_null = float(row.isnull().sum()) / row.count()
    return pct_null < 0.8

df = df[df.apply(is_poor_coverage, axis=1)]
df['year_issued'] = df.issue_d.apply(lambda x: int(x.split("-")[0]))
df_term = df[df.year_issued < 2012]
bad_indicators = [
    "Late (16-30 days)",
    "Late (31-120 days)",
    "Default",
    "Charged Off"
    ]

df_term['is_rent'] = df_term.home_ownership=="RENT"
df_term['is_bad'] = df_term.loan_status.apply(lambda x: x in bad_indicators)
features = ['last_fico_range_low', 'last_fico_range_high', 'is_rent']
clf = LogisticRegression()
model = clf.fit(df_term[features], df_term.is_bad)
joblib.dump(model, 'creditscore.pkl')
