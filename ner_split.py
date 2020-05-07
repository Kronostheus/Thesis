import pandas as pd
import glob
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from string import punctuation, ascii_letters, digits
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizerFast
from random import shuffle
from tqdm import tqdm

CLASS_DIR = "Data/Classification/"
DATA_DIR = "Data/"

TRAIN_DIR = CLASS_DIR + 'Train/'
VAL_DIR = CLASS_DIR + 'Val/'
TEST_DIR = CLASS_DIR + 'Test/'

PORTUGAL_DATA = glob.glob(DATA_DIR + '/Portugal_Manifestos/*.csv')
BRAZIL_DATA = glob.glob(DATA_DIR + '/Brazil_Manifestos/*.csv')
ITALY_DATA = glob.glob(DATA_DIR + '/Italy_Manifestos/*.csv')

all_data = PORTUGAL_DATA + BRAZIL_DATA + ITALY_DATA

df_list = [pd.read_csv(path) for path in all_data]
shuffle(df_list)


def capitalize(match):
    return match.group(1).upper()


def tokenize_mask(span_length, is_span):
    return ['B'] + ['I'] * (span_length - 1) if is_span else ['O'] * span_length


def new_match_mask(sentences, masks):
    new_masks = []
    for sent in sentences:
        new_masks.append(masks[:len(sent)].strip())
        masks = masks[len(sent)+1:].strip()
    return new_masks


def fix_chunk_mask(masks, codes):

    def incorporate_right(split):
        return " ".join(split[:-1]) + split[-2][-1] * (len(split[-1]) + 1)

    new_masks, new_codes = [], []

    for mask, code in zip(masks, codes):
        tmp_mask = mask
        tmp_code = code

        mask_spans = tmp_mask.split()
        code_spans = tmp_code.split()

        # Propagate changes to code
        if mask[0] == 'I':
            # Span got cut in the middle and next mask begins on an Inside
            if len(mask_spans[0]) > 1:
                tmp_mask = "B" + mask[1:]
            # Lone 'I' (ex: I BIII BIIIII) that will be included in the same span
            elif len(mask_spans) > 1:
                first_token = mask_spans[1][0]
                fill_tokens = "I" if first_token == "B" else "O"
                tmp_mask = first_token + fill_tokens * 2 + mask[3:]
                tmp_code = code_spans[1][0] * 2 + tmp_code[2:]

        mask_spans = tmp_mask.split()
        code_spans = tmp_code.split()

        # Single BIO tag at end, probably cut during chunking.
        if len(mask_spans) > 1 and len(mask_spans[-1]) < 2 and mask_spans[-1][0] != mask_spans[-2][-1]:
            tmp_mask = incorporate_right(mask_spans)
            tmp_code = incorporate_right(code_spans)

        new_masks.append(tmp_mask)
        new_codes.append(tmp_code)

    return new_masks, new_codes


def sentence_fix(sentences, masks, codes):
    new_sentences, new_masks, new_codes = [], [], []
    tmp_sentence, tmp_mask, tmp_code = sentences[0], masks[0], codes[0]

    for i, (sentence, mask, code) in enumerate(zip(sentences, masks, codes)):
        if mask[0] in ('B', 'O'):
            new_sentences.append(tmp_sentence)
            new_masks.append(tmp_mask)
            new_codes.append(tmp_code)
            tmp_sentence, tmp_mask, tmp_code = sentence, mask, code
        else:
            tmp_sentence = '{} {}'.format(tmp_sentence, sentence)
            tmp_mask = '{}I{}'.format(tmp_mask, mask)
            tmp_code = '{}{}{}'.format(tmp_code, code[0], code)

    return new_sentences[1:], new_masks[1:], new_codes[1:]


def chunk_sentence(bert, sentence):
    sentence_chunks = ["@"]
    split_sentence = sentence.split()
    max_length = 100

    while split_sentence:
        tmp_chunk = split_sentence[:max_length + 1]

        commas = [idx for idx, split in enumerate(tmp_chunk) if split[-1] == ',']

        if len(split_sentence) > 100 or len(sentence_chunks) == 1:
            last_comma = commas[-1] if commas else int((len(tmp_chunk) - 1) * 0.75)
        else:
            last_comma = len(tmp_chunk) - 1

        tmp_chunk = " ".join(split_sentence[:last_comma + 1])

        if len(bert.encode(tmp_chunk)) < 200:
            sentence_chunks.append(tmp_chunk)
            split_sentence = split_sentence[last_comma + 1:]
            max_length = 100
        else:
            max_length = last_comma - 1

    return sentence_chunks[1:]


def chunk_sentences(bert, sentences, masks, codes):
    new_sentences, new_masks, new_codes = [], [], []

    for i, (sentence, mask, code) in enumerate(zip(sentences, masks, codes)):
        if len(bert.encode(sentence)) > 200:

            chunked_sentences = chunk_sentence(bert, sentence)

            tmp_masks = new_match_mask(chunked_sentences, mask)
            tmp_codes = new_match_mask(chunked_sentences, code)

            chunked_masks, chunked_codes = fix_chunk_mask(tmp_masks, tmp_codes)

            new_sentences.extend(chunked_sentences)
            new_masks.extend(chunked_masks)
            new_codes.extend(chunked_codes)
        else:
            new_sentences.append(sentence)
            new_masks.append(mask)
            new_codes.append(code)

    return new_sentences, new_masks, new_codes


def tokenize_sentence(sentences, masks, codes):

    result = {"sentence_id": [], "words": [], "labels": [], "codes": []}

    for i, (sentence, mask, code) in enumerate(zip(sentences, masks, codes)):

        # Get span limits through its mask
        span_seps = [pos for pos, char in enumerate(mask.strip()) if char == ' ']

        spans = []
        cursor = -1
        for sep in span_seps:
            """
            Break sentence with multiple spans into a list of its respective spans.
            """
            spans.append(sentence[cursor + 1:sep].strip())
            cursor = sep
        spans.append(sentence[cursor + 1:].strip())

        span_masks = mask.split()
        span_codes = code.split()

        tokenized_spans, tokenized_masks, tokenized_codes = [], [], []
        for span, span_mask, span_code in zip(spans, span_masks, span_codes):
            """
            Tokenize everything according to nltk's word_tokenizer
            """
            words = word_tokenize(span)
            tokenized_spans.extend(words)
            tokenized_codes.extend([span_code[0]] * len(words))
            tokenized_masks.extend(tokenize_mask(len(words), span_mask[0] in ('B', 'I')))

        if len(tokenized_spans) != len(tokenized_masks):
            breakpoint()
            return {"sentence_id": [], "words": [], "labels": [], "codes": []}

        result["sentence_id"].extend([i] * len(tokenized_spans))
        result["words"].extend(tokenized_spans)
        result["labels"].extend(tokenized_masks)
        result["codes"].extend(tokenized_codes)

    return result


def fix_sentence_id(dfs):
    offset = 0
    for df in dfs:
        df.sentence_id += offset
        offset += df.shape[0]
    return dfs


def split_dfs(dfs, train_perc, test_perc):
    lens = {i: df.shape[0] for i, df in enumerate(dfs)}

    total = sum(lens.values())

    train_size = total * train_perc
    test_size = total * (train_perc + test_perc)

    cursor = 0
    seps = []
    for dfi, size in lens.items():
        cursor += size
        if cursor > train_size and not seps:
            seps.append(dfi)
        elif cursor > test_size and len(seps) == 1:
            seps.append(dfi)

    return pd.concat(dfs[:seps[0]+1]), pd.concat(dfs[seps[0]+1:seps[1]+1]), pd.concat(dfs[seps[1]+1:])


alphanum = "{}{}".format(digits, ascii_letters)

# Code to Unit
u_code = {str(code): alphanum[ic] for ic, code in enumerate(pd.concat(df_list).Code.unique())}

# Unit to Code
inv_u_code = {v: k for k, v in u_code.items()}

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

token_dfs = []

for manifesto_df in tqdm(df_list, desc="Processing DataFrames: "):

    manifesto_text = ""
    masked_text = ""
    masked_code = ""

    for index, row in manifesto_df.iterrows():
        text = " ".join(sent_tokenize(" ".join(row.Text.split())))
        text = text.replace('\uf020', '').replace('//', '--').replace(';.', '; ')
        code = str(row.Code)

        bio = "B" if code not in ('000', '0', '999') else 'O'
        mask_in = "I" if bio == 'B' else 'O'
        text_mask = bio + mask_in * (len(text) - 1)

        code_mask = u_code[code] * len(text)

        masked_text += " {}".format(text_mask)
        manifesto_text += " {}".format(text)
        masked_code += " {}".format(code_mask)

    manifesto_text = manifesto_text.strip()
    masked_text = masked_text.strip()
    masked_code = masked_code.strip()

    manifesto_text = re.sub('(; [a-z0-9])', capitalize, manifesto_text) # capitalize first letter after semi-colon
    manifesto_text = manifesto_text.replace(';', '.')
    sent_list = sent_tokenize(manifesto_text)

    mask_list = new_match_mask(sent_list, masked_text)
    code_list = new_match_mask(sent_list, masked_code)

    s_list, m_list, c_list = sent_list, mask_list, code_list

    sent_list, mask_list, code_list = sentence_fix(sent_list, mask_list, code_list)

    sent_list, mask_list, code_list = chunk_sentences(tokenizer, sent_list, mask_list, code_list)

    tokens = tokenize_sentence(sent_list, mask_list, code_list)

    data = pd.DataFrame(tokens)

    data.codes = data.codes.apply(lambda x: inv_u_code[x])
    token_dfs.append(data)

token_dfs = fix_sentence_id(token_dfs)

train, test, val = split_dfs(token_dfs, 0.7, 0.15)

print("Train:{}\nTest:{}\nVal:{}".format(train.shape[0], test.shape[0], val.shape[0]))

train.to_csv(TRAIN_DIR + "train_ner.csv", index=False)
val.to_csv(VAL_DIR + 'val_ner.csv', index=False)
test.to_csv(TEST_DIR + 'test_ner.csv', index=False)
