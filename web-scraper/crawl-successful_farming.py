import requests
import time
import math
from bs4 import BeautifulSoup
import os
import json
from tqdm import tqdm

"""
pre-determined KEYWORDS dict = {<query>: <total no. of results>}
"""

KEYWORDS = {'wheat': 8924, 'corn': 15009, 'oats': 476, 'rice': 999}

OUTPUT_PATH = './successful_farming/'

def get_raw_html(query, page):
    url = f'https://www.agriculture.com/search?search_api_views_fulltext={query}&sort_by=created&page={page}'
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
    soup = BeautifulSoup(raw_html)
    raw_articles = soup.find_all('article')
    url = [r.find('h2').find('a').get('href') for r in raw_articles]
    headline = [r.find('h2').text for r in raw_articles]
    # body = [[s.text for s in r.find('div', class_='field-body').find_all('p')] for r in raw_articles]
    footer = [r.find('footer').text for r in raw_articles]
    result = [{'headline':h, 'metadata':f, 'url': u, 'query':query} for h,f,u in zip(headline,footer,url)]
    return result


if __name__ == '__main__':

    for query, n in KEYWORDS.items():
        for page in tqdm(range(math.ceil(n/10))):
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