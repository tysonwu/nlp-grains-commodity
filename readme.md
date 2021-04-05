# Notes to Files Related to this Project

This folder contains several Python files and Jupyter notebooks as a demonstration for the main project.

## Report File

The `report` folder contains the main Word document of the project report.

## Presentation slides

The `presentation_slides` folder contains the PDF document of the presentation slides.

## Programs

The main programmes are stored under the `codes` folder which contains several components.

### `data` folder

- The crawling result data are stored in `news-data-financial_post` which contains result from crawling on the news website Financial Post, `news-data-successful_farming` which contains result from crawling on the news website Sucessful Farming, and `twitter-data` contains result from Twitter crawling.
- The data in the above data folders are not complete, as the complete set of data is too large to contain.

### `web-scraper` folder

- The codes for crawling and cleaning the news datafrom news webiste are stored according to source news website, in the `crawl-successful_farming.py ` and `crawl-financial_post.py` file. The program outputs the crawled data in JSON format.
- The engine to crawl `twitter-data` is not written by us. Credits to an online GitHub repo `twitterscraper` by `@jonbakerfish` (https://github.com/jonbakerfish/TweetScraper). The source code is also stored under the folder `web-crawler` for reference.
- The codes for using the mentioned Twitter engine and cleaning of Twitter data are stored in `clean-twitter-data.py` file.
- Separate Jupyter notebooks are provided as a guide of walking through the programs.

### `lexicon-construction` folder

- The codes for constructing the lexicon database using the crawled data are stored in the `lexicon_construction.py` file. The program outputs `lexicon_scores.json` as result.
- Separate Jupyter notebooks are provided as a guide of walking through the program.

### `price-analysis` folder

- The codes for conducting statistical analysis using the data and lexicon dictionaries are stored in `price-analysis.py` file.
- Separate Jupyter notebooks are provided as a guide of walking through the program.

### `output` folder

- The output folder contains data of the lexicons, with their respective sentiment values stored in the file `lexicon_scores.json`

