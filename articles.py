import pandas as pd
import re
from nltk import sent_tokenize, word_tokenize
from string import punctuation
from Utils.config import Config
from pathlib import Path

data = pd.read_json(Config.OPINION_DIR + 'dados_artigos.json')
data = data[~data.url.str.contains('arquivo')]
data = data[~data.title.duplicated()]


def build_article_text(article, dirname):
    # Clean text by removing excessive whitespace and remove enumerations (ex: '1. One item' -> 'One item')
    body = " ".join(re.sub(r'^\d+\.\s', '', text) for text in article.body.strip().split())

    # Remove punctuation from title which could through errors when naming file
    with open(file=dirname + str(article.title).translate(str.maketrans('', '', punctuation)) + '.txt',
              mode='w',
              encoding='utf-8') as file:

        file.writelines("\n".join(sent_tokenize(body)))


def build_article_df(article_num, body):
    df = pd.DataFrame(
        sent_tokenize(
            " ".join(re.sub(r'^\d+\.\s', '', text) for text in body.strip().split())
        ),
        columns=['sentences']
    )

    df['article'] = article_num
    return df


all_sentences = []

for author, author_df in data.groupby(by='author'):

    dirname = Config.OPINION_DIR + author + '/'
    Path(dirname).mkdir(exist_ok=True)
    # author_df.apply(lambda row: build_article_text(row, dirname), axis=1)
    sentences_df = pd.concat(build_article_df(article, body) for article, body in enumerate(author_df.body))
    sentences_df['author'] = author
    all_sentences.append(sentences_df)

all_df = pd.concat(all_sentences).reset_index(drop=True)
breakpoint()