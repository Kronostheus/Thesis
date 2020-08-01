import pandas as pd
import random
import datetime
from googletrans import Translator
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import namedtuple
from nltk.tokenize import word_tokenize
from Utils import Config, get_spans


# Each language is translated into all the others
dests = {val: list(set([dst for dst in Config.langs.values() if dst != val])) for val in Config.langs.values()}

train_ner = pd.read_csv(Config.TRAIN_DIR + 'train_ner.csv', header=0)

sent_span = namedtuple('Row', 'span relevant code origin')

translator = Translator()


def tokens(length, is_relevant):
    return ['B'] + ['I'] * (length - 1) if is_relevant else ['O'] * length


def flatten(lst):
    tokens_lst, labels_lst = [], []
    for span_lst in lst:
        tokens_lst.extend(span_lst[0])
        labels_lst.extend(span_lst[1])
    assert len(tokens_lst) == len(labels_lst)
    return [tokens_lst, labels_lst]


def translate(sent):

    lang1, lang2, lang3 = [], [], []
    for row in sent:
        dest_langs = dests[row.origin]

        t1 = word_tokenize(translator.translate(row.span, src=row.origin, dest=dest_langs[0]).text)
        t1_list = [t1, tokens(len(t1), row.relevant)]
        lang1.append(t1_list)

        t2 = word_tokenize(translator.translate(row.span, src=row.origin, dest=dest_langs[1]).text)
        t2_list = [t2, tokens(len(t2), row.relevant)]
        lang2.append(t2_list)

        t3 = word_tokenize(row.span)
        t3_list = [t3, tokens(len(t3), row.relevant)]
        lang3.append(t3_list)

    return [flatten(lang) for lang in (lang1, lang2, lang3)]


sentences = []
t = []
for _, sent_df in train_ner.groupby(by='sentence_id'):

    sentence = []

    spans = get_spans(sent_df.labels, sent_df.codes)

    country = Config.langs[sent_df.country.unique()[0]]

    for start, end in zip(spans, spans[1:]):

        span = " ".join(sent_df.words.astype(str)[start:end])
        code = sent_df[start:end].codes.unique()[0]
        relevant = False if 'O' in sent_df[start:end].labels.values else True

        sentence.append(sent_span(span, relevant, code, country))

    sentences.append(sentence)


row_lst = []

print(datetime.datetime.now())

with ThreadPoolExecutor(max_workers=20) as executor:
    row_lst.extend(
        executor.map(translate, sentences)
    )

print(datetime.datetime.now())

random.seed(42)

row_lst = [r for trsl in row_lst for r in trsl]

random.shuffle(row_lst)

sent_id = 0
ids, words, labels = [], [], []
for translation in tqdm(row_lst):
    ids.extend([sent_id] * len(translation[0]))
    words.extend(translation[0])
    labels.extend(translation[1])
    sent_id += 1

data = pd.DataFrame(zip(ids, words, labels), columns=['sentence_id', 'words', 'labels'])
data.to_csv('Data/Classification/Train/train_ner_translated.csv', index=False)
