import pandas as pd
import re
import datetime
from googletrans import Translator
from collections import namedtuple
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from Utils import Config, get_spans_old


train_ner = pd.read_csv(Config.CLASS_DIR + 'Train/train_ner.csv')
train_translated = pd.read_csv(Config.CLASS_DIR + 'Train/train_translated.csv')

# Each language is translated into all the others
dests = {val: list(set([dst for dst in Config.langs.values() if dst != val])) for val in Config.langs.values()}

tmp_spans = namedtuple('Row', 'spans code origin')

translator = Translator()


def translate(row):
    dest_langs = dests[row.origin]
    translated_spans = []

    for span in row.spans:
        t1 = translator.translate(span, src=row.origin, dest=dest_langs[0]).text
        t2 = translator.translate(span, src=row.origin, dest=dest_langs[1]).text
        translated_spans.append([span, t1, t2])

    translated_spans = [" ".join(combination) for combination in product(*translated_spans)]
    return list(zip(translated_spans, [row.code for _ in translated_spans]))


"""
Acquire data from train_ner.csv
    - Grab sentences with multiple (>1) spans with the exact same Code
    - Minimum amount of spans = 2
    - Maximum amount of spans = 5
"""

to_translate = []

for _, sent_df in train_ner.groupby(by='sentence_id'):

    spans = get_spans_old(sent_df.labels, sent_df.codes)

    # 1 span -> len = 2 | 5 spans -> len = 6
    if len(spans) in range(3, 7) and len(sent_df.codes.unique()) == 1:

        rebuilt_spans = []

        # Rebuild span. Regular expression in order to better position punctuation in resulting spans.
        for start, end in zip(spans, spans[1:]):
            rebuilt_spans.append(re.sub(r'\s+([?.!,:])', r'\1', " ".join(sent_df.words.astype(str)[start:end])))

        to_translate.append(tmp_spans(
            rebuilt_spans,
            sent_df.codes.unique()[0],
            Config.langs[sent_df.country.unique()[0]]
        ))

"""
Process acquired sentences
    - Distributed processing
    - Knowing the origin language of the sentence, translate its spans into remaining languages
    - All possible language combinations are produced as long as span order is kept
    - Amount of resulting spans per sentence = 3 ** #Spans
    - Total amount of resulting spans is around 50K
"""

row_lst = []

print(datetime.datetime.now())

with ThreadPoolExecutor(max_workers=20) as executor:
    row_lst.extend(
        executor.map(translate, to_translate)
    )

print(datetime.datetime.now())

row_lst = [r for trsl in row_lst for r in trsl]

translated_data = pd.DataFrame(row_lst, columns=['Text', 'Code'])
train_translated = train_translated.append(translated_data, ignore_index=True).sample(frac=1, random_state=42)
train_translated.to_csv(Config.TRAIN_DIR + 'train_combinations.csv', index=False)
