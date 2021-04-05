import os
import glob
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from dateutil.parser import parse
import re


def preprocessing(x):
    x = x.lower() # converts to lower case
    x = re.sub(r'http:[^\ ]*','',x) # exclude websites
    x = re.sub(r'www[^\ ]*','',x)
    x = re.sub(r'@[^\ ]*','',x)
    x = re.sub(r'[/@#:\)\/\n\_\-\&]','',x)
    
    def deEmojify(text):
        regrex_pattern = re.compile(pattern = "["u"\U0001F600-\U0001F64F"u"\U0001F300-\U0001F5FF"u"\U0001F680-\U0001F6FF"u"\U0001F1E0-\U0001F1FF""]+", flags = re.UNICODE)
        return regrex_pattern.sub(r'',text)
    
    x = deEmojify(x)
    return x


if __name__ == '__main__':
    fn = glob.glob('./tweetscraper/Data/tweet/*')
    tweets = []
    for fname in tqdm(fn):
        with open(fname, 'r') as f:
            data = json.load(f)
            tweets.append({'date':data['raw_data']['created_at'], 
                        'headline': data['raw_data']['full_text'],
                        'hashtags': [i['text'] for i in data['raw_data']['entities']['hashtags']],
                        'source': 'tweets',
                        'query': 'grains'})

    df = pd.json_normalize(tweets)
    df['date'] = df['date'].apply(parse).apply(lambda x: x.date())
    df['url'] = None
    df['type'] = 'tweets'
    df['headline'] = df['headline'].apply(preprocessing)
    df = df.sort_values('date')
    df.to_csv('tweets.csv',index=False)