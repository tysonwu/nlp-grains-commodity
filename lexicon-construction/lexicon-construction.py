from datetime import datetime
from dateutil.parser import parse
import pandas as pd
import numpy as np
import json
import pysentiment2 as ps
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import en_core_web_sm
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import Counter
from tqdm import tqdm


nlp = en_core_web_sm.load()
stop_words = set(stopwords.words('english'))
sentiment_levels = [-1, 0, 1]

KEYWORDS = ['corn','oats','wheat','soybean','rice']
QUERIES = ['corn','oats','wheat','soybean','rice']
OUTPUT_PATH = './successful_farming/'


def load_data_sf():
    temp = []
    for query in QUERIES:
        with open(f'./successful_farming/{query}.json','r') as f:
            data = json.load(f)
            temp += data['data']
    df = pd.json_normalize(temp)
    return df
        

def load_data_fp():
    temp = []
    for query in QUERIES:
        with open(f'./financial_post/{query}.json','r') as f:
            data = json.load(f)
            temp += data['data']
    df = pd.json_normalize(temp)
    return df
        
    
    
def clean_sf(df):
    df = df.drop_duplicates()
    df['date'] = df['metadata'].apply(lambda x: datetime.strptime(x.split('|')[0].strip(),'%m.%d.%Y'))
    df['type'] = df['metadata'].apply(lambda x: x.split('|')[1].strip())
    del df['metadata']
    df = df[df['type']=='Article']
    
    # remove '3 big things today', 'Three commodities to watch'
    df = df[~df['headline'].str.contains('3 Big Things Today')]
    df = df[~df['headline'].str.contains('Three commodities to watch')]
    
    return df


def clean_fp(df):
    df = df.drop_duplicates()
    df = df[~df['metadata'].str.contains('ago')]
    df['date'] = df['metadata'].apply(parse)
    df = df[df['cat'].isin(['PMN Business','Agriculture','Business','PMN Agriculture'])]
    df['type'] = df['cat']
    del df['metadata']
    del df['cat']
    
    return df


def generate_sentiment(df):

    # using textblob for polarity
    # using LM dictionary
    # using HIV4 dictionary
    # using VADER

    lm = ps.LM()
    hiv4 = ps.HIV4()
    vader = SentimentIntensityAnalyzer()
    
    # VADER score
    df['VADER_compound'] = df['headline'].apply(lambda x: vader.polarity_scores(x)['compound'])

    # Harvard IV-4 dictionary - HIV4 score 
    df['HIV4_score'] = df['headline'].apply(lambda x: hiv4.get_score(hiv4.tokenize(x))['Polarity'])
    
    # Loughran and McDonald Financial Sentiment Dictionary - LM score
    df['LM_score'] = df['headline'].apply(lambda x: lm.get_score(lm.tokenize(x))['Polarity'])
    
    # TextBlob polarity - identifies negation
    df['TextBlob_polarity'] = df['headline'].apply(lambda x: TextBlob(x).sentiment[0])

    # TextBlob subjectivity
    df['TextBlob_subjectivity'] = df['headline'].apply(lambda x: TextBlob(x).sentiment[1])
    
    df = df.drop_duplicates(subset='headline')
    
    del df['url']
    del df['type']
    
    return df


def tokenize(h):
    tokens = word_tokenize(h)
    words = [word.lower() for word in tokens if word.isalpha()]
    words = [word for word in words if not word in stop_words]
    
    # stemming
    ps = PorterStemmer()
    words = [ps.stem(w) for w in words]

    # lemmatization
    Lemmatizer = WordNetLemmatizer()
    words = [Lemmatizer.lemmatize(w, pos='v') for w in words]
    
    return ' '.join(words)


def ner(h):
    h = re.sub(r'[^a-zA-Z ]*','',h).lower() # sub all except alphabets
    sen = nlp(h)
    return sen.ents


def classify_sentiment(x):

    # cutoff for neg/neu/pos classification: -0.25/0.25

    if x <= -0.25:
        return -1
    elif x >= 0.25:
        return 1
    else:
        return 0


def convert_corpus_to_lexicon(df, sl):
    corpus = [w.split(' ') for w in df[df['sentiment']==sl]['headline_tk'].to_list()]
    corpus = [item for sublist in corpus for item in sublist] # flatten
    
    corpus_threshold = sum([v for v in dict(Counter(corpus)).values()]) * 0.00005
    lexicon = [k for k,v in dict(Counter(corpus)).items() if v>=corpus_threshold]
    lexicon = [word for word in lexicon if word not in ents]
    
    # remove stopwords from lexicon and corpus
    lexicon = [word for word in lexicon if word not in stop_words]
    corpus = [word for word in corpus if word not in stop_words]
    return lexicon, corpus


def generate_lexicon_score(h):
    score = 0
    for word in h.split(' '):
        if word in lexicon_scores.keys():
            score += lexicon_scores[word]
    return score


if __name__ == '__main__':
    df_fp = load_data_fp()
    df_fp = clean_fp(df_fp)

    df_sf = load_data_sf()
    df_sf = clean_sf(df_sf)

    df = pd.concat([df_fp,df_sf])
    df['hashtag'] = None
    df['source'] = 'news'
    df.head()

    # concat with Twitter data after the colnames are uniformized
    df_tweets = pd.read_csv('./tweets.csv',lineterminator='\n')
    df = pd.concat([df, df_tweets])

    df = generate_sentiment(df)

    df['headline_tk'] = df['headline'].apply(tokenize)
    df['ents'] = df['headline'].apply(ner)

    # generate a compound score for overall sentiment classification

    df['total_score'] = df['HIV4_score']*0.5+df['LM_score']*0.5
    df['HIV4_LM_agreement'] = np.where(abs(df['HIV4_score']-df['LM_score'])>0.8, -1, 1)
    df['overall_polarity'] = df['TextBlob_polarity'].apply(lambda x: 1 if x>=0 else -1)
    df['polarized_score'] = np.where(df['HIV4_LM_agreement']==1, df['total_score'], abs(df['total_score'])*df['overall_polarity'])
    
    df['sentiment'] = df['polarized_score'].apply(classify_sentiment)

    lexicons = {}
    corpuses = {}
    for sl in sentiment_levels:
        lexicons[sl], corpuses[sl] = convert_corpus_to_lexicon(df, sl=sl)


    total_lexicons = [corpuses[sl] for sl in sentiment_levels]
    total_lexicons = [item for sublist in total_lexicons for item in sublist if item != ''] # flattten
    lexicon_set = list(set(total_lexicons))

    # Pointwise mutual information (PMI) calculation
    """
    PMI(w,c) = log( p(w,c) / p(w) / p(c) )
    representing the confidence of sentiment of word
    then overall score for w is PMI(w,1) - PMI(w,-1)
    """
    # calculate p(c)
    p_c = {}
    for sl in tqdm(sentiment_levels):
        p_c[sl] = len(corpuses[sl])/len(total_lexicons)

    # calculate p(w)
    lexicon_counter = Counter(total_lexicons)
    p_w = {}
    for lex in tqdm(lexicon_set):
        p_w[lex] = lexicon_counter[lex]/len(total_lexicons)

    # calculate p(w,c)
    p_wc = {}
    pmi = {}

    counter_courpuses = {}

    for sl in tqdm(sentiment_levels):
        counter_courpuses[sl] = Counter(corpuses[sl])

    for lex in tqdm(lexicon_set):
        for sl in sentiment_levels:
            p_wc[(lex, sl)] = counter_courpuses[sl][lex] / len(total_lexicons)
            pmi[(lex, sl)] = np.log(p_wc[(lex, sl)]/p_c[sl]/p_w[lex]) if p_wc[(lex, sl)] != 0 else 0

    lexicon_scores = {}
    for lex in lexicon_set:
        lexicon_scores[lex] = pmi[(lex, 1)] - pmi[(lex, -1)]

    pmi_max = max(list(lexicon_scores.values()))
    pmi_min = min(list(lexicon_scores.values()))

    lexicon_scores = {k:v for k, v in lexicon_scores.items() if k not in ents}

    df['lexicon_score'] = df['headline_tk'].apply(generate_lexicon_score)
    df = df.sort_values('lexicon_score',ascending=False)

    with open('./lexicon_scores.json', 'w+') as f:
        json.dump(lexicon_scores,f)