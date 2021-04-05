import requests
import time
import math
from bs4 import BeautifulSoup
import os
import json
from tqdm import tqdm


def get_raw_html(query, count):
    url = f'https://financialpost.com/search/?search_text={query}&date_range=-7300d&sort=desc&from={count}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2490.80 Safari/537.36',
        'Content-Type': 'text/html',
    }
    contents = requests.get(url,headers=headers).text
    time.sleep(0.5)
#     print(f'Scraped one page: {query} - {page}')
    return contents


def html_transform(raw_html, query):
    """
    arg raw_html: raw html string
    return: a list of dict
    """
    soup = BeautifulSoup(raw_html,'lxml')
    raw_articles = soup.find_all('article')
    url = [r.find('a', class_='article-card__link').get('href') for r in raw_articles]
    headline = [r.find('h3').text for r in raw_articles]
    # body = [[s.text for s in r.find('div', class_='field-body').find_all('p')] for r in raw_articles]
    footer = [r.find('span', class_='article-card__time').text for r in raw_articles]
    cat = [r.find('span', class_='article-card__category').text for r in raw_articles]
    result = [{'headline':h, 'metadata':f, 'url': u, 'cat': c, 'query':query} for h,f,u,c in zip(headline,footer,url,cat)]
    return result


if __name__ == '__main__':

for query, n in KEYWORDS.items():
    """
    pre-determined KEYWORDS dict = {<query>: <total no. of results>}
    """
    KEYWORDS = {'soybean': 2963, 'wheat': 4476, 'corn': 4041, 'oats': 863, 'rice': 1480}
    OUTPUT_PATH = './financial_post/'
    
    for page in tqdm(range(0,math.ceil(n/10)*10,10)):
        raw_htmls = get_raw_html(query,page) # returns a string of raw html
        result = html_transform(raw_htmls, query)

        # writing to json file
        if not os.path.isfile(f'{OUTPUT_PATH}{query}.json'): # if file is empty
            with open(f'{OUTPUT_PATH}{query}.json','w+') as f:
                json.dump({'data': result}, f)
        else:
            with open(f'{OUTPUT_PATH}{query}.json','r') as f:
                data = json.load(f)
            data['data'] += result
            with open(f'{OUTPUT_PATH}{query}.json','w') as f:
                json.dump(data,f)
