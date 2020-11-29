import re
import pandas as pd
from tqdm import tqdm


def get_spans_old(labels, codes):
    limits = []
    prev = ""
    for idx, (lbl, code) in enumerate(zip(labels, codes)):
        if lbl == 'B' or (prev != code and lbl == 'O'):
            limits.append(idx)
        prev = code

    limits.append(len(labels))

    if len(limits) <= 1:
        if limits[0] == 0:
            raise Exception
        else:
            limits = [0] + limits
    elif limits[0] != 0:
        limits = [0] + limits

    return limits


def get_spans(labels):
    return [match.span() for match in re.finditer(r'BI*', "".join(labels))]


def build_span_df(id_df):
    rows = []
    for _, sent_df in tqdm(id_df.groupby(by='sentence_id')):

        true_spans = get_spans_old(sent_df.labels.to_list(), sent_df.codes.to_list())

        for start, end in zip(true_spans, true_spans[1:]):
            df = sent_df[start:end]
            rows.append([
                " ".join(str(word) for word in df.words),
                df.codes.unique()[0],
                df.country.unique()[0]
            ])
    return pd.DataFrame(rows, columns=['Text', 'Code', 'country'])
