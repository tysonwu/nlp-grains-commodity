import requests
import time
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm


KEYWORDS = ['corn','oats','rice','soybean','wheat','grains']

# using CME price
api_params = {
    'corn': 'CHRIS/CME_C1',
    'oats': 'CHRIS/CME_O1',
    'rice': 'CHRIS/CME_RR1',
    'soybean': 'CHRIS/CME_S1',
    'wheat': 'CHRIS/CME_W1'
}


def get_price(param):
    
    url = f'http://www.quandl.com/api/v3/datasets/{param}'
    r = requests.get(url).json()
    time.sleep(0.5)
    colnames = r['dataset']['column_names']
    data = r['dataset']['data']
    df = pd.DataFrame(data, columns=colnames)
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df = df.sort_values('Date')
    df['logreturn'] = np.log(df['Last']/df['Last'].shift(1))
    
    df = df.reset_index()
    df = df.set_index('Date')
    
    return df


def load_data():
    df = pd.read_csv('./dataframe.csv',lineterminator='\n')
    return df


def preprocessing(df):
    df = df[['date','query','lexicon_score']]
    df['date'] = df['date'].apply(lambda x: x[:11].strip())
    df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    
    dfs = {}
    for key in KEYWORDS:
        # aggregate data by time
        temp = df[df['query']==key]
        dfs[key] = temp[['date','lexicon_score']].groupby('date').mean()
        
    return dfs


if __name__ == '__main__':
    prices = {}

    for k, v in api_params.items():
        prices[k] = get_price(v)


    df = load_data()
    dfs = preprocessing(df)


    for query in KEYWORDS:
        print(query)

        dfm = dfs[query].join(prices[query]['logreturn'], how='left')
        dfm = dfm.resample('2M').agg({'lexicon_score':'mean','logreturn':'sum'})
        dfm = dfm[dfm.index >= datetime(2017,1,1)]

        for n in range(1, 7):
            dfm[f'lexicon_score_lag_{n}'] = dfm['lexicon_score'].shift(n)

        dfm = dfm.dropna()

        X = dfm.drop(['logreturn','lexicon_score'], axis=1)
        Y = dfm[['logreturn']]

        X = sm.add_constant(X) # adding a constant

        model = sm.OLS(Y, X).fit()
        print_model = model.summary()
        print(print_model)
        print('###################################################')


    for query in KEYWORDS:
        print(query)
        dfm = dfs['grains'].join(prices[query]['logreturn'], how='left')
        dfm = dfm.resample('4D').agg({'lexicon_score':'mean','logreturn':'sum'})
        dfm = dfm[dfm.index >= datetime(2017,1,1)]

        for n in range(1, 7):
            dfm[f'lexicon_score_lag_{n}'] = dfm['lexicon_score'].shift(n)

        dfm = dfm.dropna()

        X = dfm.drop(['logreturn','lexicon_score'], axis=1)
        Y = dfm[['logreturn']]

        X = sm.add_constant(X) # adding a constant

        model = sm.OLS(Y, X).fit()
        print_model = model.summary()
        print(print_model)
        print('###################################################')