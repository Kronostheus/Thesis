import pandas as pd
import glob

from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation
from random import shuffle
from tqdm import tqdm

CLASS_DIR = "Data/Classification/"
DATA_DIR = "Data/"

TRAIN_DIR = CLASS_DIR + 'Train/'
VAL_DIR = CLASS_DIR + 'Val/'
TEST_DIR = CLASS_DIR + 'Test/'

PORTUGAL_DATA = glob.glob(DATA_DIR + '/Portugal_Manifestos/*.csv')
BRAZIL_DATA = glob.glob(DATA_DIR + '/Brazil_Manifestos/*.csv')

all_data = PORTUGAL_DATA + BRAZIL_DATA

df_list = [pd.read_csv(path) for path in all_data]
shuffle(df_list)


def match_mask(sentences, masks):
    rejoin = "@".join(sentences)
    span_seps = [pos for pos, char in enumerate(rejoin) if char == '@']
    return [masks[:span_seps[0]]] + [masks[start + 1: end] for start, end in zip(span_seps, span_seps[1:])] + [masks[span_seps[-1] + 1:]]


def tokenize_mask(span_length, is_span):
    return ['B'] + ['I'] * (span_length - 1) if is_span else ['O'] * span_length


def tokenize_sentence(sentences, masks):

    result = {"sentence_id": [], "words": [], "labels": []}

    for i, (sentence, mask) in enumerate(zip(sentences, masks)):

        span_seps = [pos for pos, char in enumerate(mask.strip()) if char == ' ']

        for j, sep in enumerate(span_seps):
            if j + 1 < len(span_seps) and span_seps[j+1] == sep + 1:
                del span_seps[j+1]

        spans = []
        cursor = -1
        for sep in span_seps:
            spans.append(sentence[cursor + 1:sep].strip())
            cursor = sep
        spans.append(sentence[cursor + 1:].strip())

        span_masks = mask.split()

        tokenized_spans, tokenized_masks = [], []
        for span, span_mask in zip(spans, span_masks):
            words = word_tokenize(span)
            tokenized_spans.extend(words)
            tokenized_masks.extend(tokenize_mask(len(words), span_mask[0] in ('B', 'I')))

        if len(tokenized_spans) != len(tokenized_masks):
            breakpoint()
            return {"sentence_id": [], "words": [], "labels": []}

        result["sentence_id"].extend([i] * len(tokenized_spans))
        result["words"].extend(tokenized_spans)
        result["labels"].extend(tokenized_masks)

    return result


def fix_sentences(sentences, masks):
    for i, (sentence, mask) in enumerate(zip(sentences, masks)):

        if len(sentence) < 5:
            if sentence in punctuation:
                sentences[i-1] += sentence
                masks[i-1] += mask
            else:
                if i + 1 >= len(sentences):
                    continue
                sentences[i+1] = sentence + ' ' + sentences[i+1]
                masks[i+1] = 'O' * len(mask) + ' ' + masks[i+1]

            del sentences[i]
            del masks[i]
            continue

        last_span = mask.split()[-1]
        if len(last_span) < 6:
            if i + 1 >= len(sentences):
                continue
            sentences[i+1] = sentence + ' ' + sentences[i+1]
            masks[i+1] = 'O' * len(mask) + ' ' + masks[i+1]
            del sentences[i]
            del masks[i]

    return sentences, masks


def fix_sentence_id(dfs):
    offset = 0
    for df in dfs:
        df.sentence_id += offset
        offset += df.shape[0]
    return dfs


def split_dfs(dfs, train_perc, test_perc):
    lens = {i: df.shape[0] for i, df in enumerate(dfs)}

    total = sum(lens.values())

    train = total * train_perc
    test = total * (train_perc + test_perc)

    cursor = 0
    seps = []
    for dfi, size in lens.items():
        cursor += size
        if cursor > train and not seps:
            seps.append(dfi)
        elif cursor > test and len(seps) == 1:
            seps.append(dfi)

    return pd.concat(dfs[:seps[0]+1]), pd.concat(dfs[seps[0]+1:seps[1]+1]), pd.concat(dfs[seps[1]+1:])


token_dfs = []

for df in tqdm(df_list, desc="Processing DataFrames: "):

    manifesto_text = ""
    masked_text = ""

    for index, row in df.iterrows():
        text = " ".join(sent_tokenize(" ".join(row.Text.split())))
        code = str(row.Code)

        bio = "B" if code not in ('000', '0', '999') else 'O'
        mask_in = "I" if bio == 'B' else 'O'
        text_mask = bio + mask_in * (len(text) - 1)

        masked_text += " {}".format(text_mask)
        manifesto_text += " {}".format(text)

    manifesto_text = manifesto_text.strip()
    masked_text = masked_text.strip()

    manifesto_text = manifesto_text.replace('//', '--').replace(';.', '.').replace(';', '.')
    sent_list = sent_tokenize(manifesto_text)

    mask_list = match_mask(sent_list, masked_text)

    sent_list, mask_list = fix_sentences(sent_list, mask_list)

    tokens = tokenize_sentence(sent_list, mask_list)
    token_dfs.append(pd.DataFrame(tokens))

token_dfs = fix_sentence_id(token_dfs)
train, test, val = split_dfs(token_dfs, 0.7, 0.15)

train.to_csv(TRAIN_DIR + "train_ner.csv", index=False)
val.to_csv(VAL_DIR + 'val_ner.csv', index=False)
test.to_csv(TEST_DIR + 'test_ner.csv', index=False)
