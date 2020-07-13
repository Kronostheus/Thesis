import pandas as pd
import glob
import re
import random

from nltk.tokenize import sent_tokenize, word_tokenize
from string import ascii_letters, digits
from transformers import BertTokenizerFast
from tqdm import tqdm

random.seed(42)

CLASS_DIR = "Data/Classification/"
DATA_DIR = "Data/"

TRAIN_DIR = CLASS_DIR + 'Train/'
VAL_DIR = CLASS_DIR + 'Val/'
TEST_DIR = CLASS_DIR + 'Test/'

all_data = {"P": glob.glob(DATA_DIR + '/Portugal_Manifestos/*.csv'),
            "B": glob.glob(DATA_DIR + '/Brazil_Manifestos/*.csv'),
            "I": glob.glob(DATA_DIR + '/Italy_Manifestos/*.csv')}

df_list = [(pd.read_csv(path), country) for country, lst in all_data.items() for path in lst]
random.shuffle(df_list)


def capitalize(match):
    """
    Capitalize regular expression group
    :param match: regular expression
    :return: capitalized regular expression
    """
    return match.group(1).upper()


def tokenize_mask(span_length, is_span):
    """
    Create a word-level BIO tokenization scheme, given a length of words and respective type
    :param span_length: How many words in span
    :param is_span: Boolean representing whether a span has a BI... or O... scheme
    :return: List with word-level BIO tokens
    """
    return ['B'] + ['I'] * (span_length - 1) if is_span else ['O'] * span_length


def match_mask(sentences, masks):
    """
    Match a mask string set to their respective sentences
    :param sentences: List of sentences
    :param masks: String of mask to be matched with sentences
    :return: List of masks segmented by the sentence limits
    """
    new_masks = []
    for sent in sentences:
        new_masks.append(masks[:len(sent)].strip())
        masks = masks[len(sent)+1:].strip()
    return new_masks


def incorporate_right(split):
    """
    Auxiliary function to fix_chunk_mask. To be used when a lone tag is found at the end of a span.
    Simply picks all masks except the last and extends the result until the limit of the span.

    ex: BIIIIII B
              ^
        BIIIIIIII

    :param split: Sentence mask split into its respective spans
    :return: Lone tag is incorporated into the preceding span.
    """
    return " ".join(split[:-1]) + split[-2][-1] * (len(split[-1]) + 1)


def fix_chunk_mask(masks, codes):
    """
    Make sure that masks are all in proper form. (ex: masks must not start with an I-tag)
    Input should match with each other, i.e. masks aligned with codes from the same sentence.
    :param masks: List[Strings] containing the tags
    :param codes: List[Strings] containing the unary code for the span category
    :return:
    """

    new_masks, new_codes = [], []

    for mask, code in zip(masks, codes):
        tmp_mask = mask
        tmp_code = code

        mask_spans = tmp_mask.split()
        code_spans = tmp_code.split()

        # Mask starts with I-tag
        if mask[0] == 'I':
            # Span got cut in the middle and next mask begins on an Inside tag
            if len(mask_spans[0]) > 1:
                # Simple put a B-tag at the beginning
                tmp_mask = "B" + mask[1:]

            # Lone 'I' (ex: I BIII BIIIII) that will be included in the same span
            elif len(mask_spans) > 1:
                # First token of proper span which will be extended
                first_token = mask_spans[1][0]

                # Tokens to use for extension
                fill_tokens = "I" if first_token == "B" else "O"

                # First token of proper span + 2 fill tokens (space + span beginning) + rest of span
                tmp_mask = first_token + fill_tokens * 2 + mask[3:]

                # Mirror above operation on code. Just take proper span's code and propagate it
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
    """
    Perform a looping action where we will iteratively obtain appropriately sized spans from text that would otherwise
    exceed the 200 BERT tokens limit. We empirically set the initial maximum amount of words at 100 as it reduces the
    need to call the BERT tokenizer.
    :param bert: BERT tokenizer
    :param sentence: Text to be chunked (already found to be >200 BERT tokens long)
    :return: List[String] containing the original sentence chunked into smaller spans of text.
    """
    # @ symbol used to ensure sentence gets chunked at least once
    sentence_chunks = ["@"]

    split_sentence = sentence.split()
    max_length = 100

    # Loop until we used all of the input text
    while split_sentence:
        # Split according to max_length of words (default = 100 words)
        tmp_chunk = split_sentence[:max_length + 1]

        # Find commas in text
        commas = [idx for idx, split in enumerate(tmp_chunk) if split[-1] == ',']

        # Check if sentence must be chunked or should we try to process it whole
        if len(split_sentence) > 100 or len(sentence_chunks) == 1:
            # Find last comma index or get 75% of the text if no comma can found
            last_comma = commas[-1] if commas else int((len(tmp_chunk) - 1) * 0.75)
        else:
            last_comma = len(tmp_chunk) - 1

        # Chunk sentence according to last comma found
        tmp_chunk = " ".join(split_sentence[:last_comma + 1])

        # Chunk is of appropriate size
        if len(bert.encode(tmp_chunk)) < 200:
            # Add chunked text to the result
            sentence_chunks.append(tmp_chunk)
            # Remove processed text
            split_sentence = split_sentence[last_comma + 1:]
            # Reset max_length
            max_length = 100
        else:
            # Shorten max_length if chunk is still too big
            max_length = last_comma - 1

    # Do not send the @ symbol (first element in list)
    return sentence_chunks[1:]


def chunk_sentences(bert, sentences, masks, codes):
    """
    Our models are limited to a maximum of 200 BERT tokens. Since we now care about the boundaries of spans, we no
    longer can simply ignore the size of the text. Therefore, we will chunk long sentences (>200 BERT tokens) into a
    series of spans, if needed.
    :param bert: BERT tokenizer
    :param sentences: List[String] of text/sentences
    :param masks: List[String] of BIO tags
    :param codes: List[String] of unary code masks for categories
    :return: All text/masks at an appropriate length
    """
    new_sentences, new_masks, new_codes = [], [], []

    for i, (sentence, mask, code) in enumerate(zip(sentences, masks, codes)):
        # Models will otherwise ignore the remaining text. We want them to classify all text/tokens.
        if len(bert.encode(sentence)) > 200:

            # Chunk long sentence
            chunked_sentences = chunk_sentence(bert, sentence)

            # Mirror changes in the masks
            tmp_masks = match_mask(chunked_sentences, mask)
            tmp_codes = match_mask(chunked_sentences, code)

            # Fix any problems associated with the chunking of the masks
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
    """
    Essentially prepares the lists in order to make them into DataFrames
    :param sentences: List[String] of sentences
    :param masks: List[String] of BIO tags
    :param codes: List[String] of unary code masks for categories
    :return: Dictionary<column_name, column values>
    """

    result = {"sentence_id": [], "words": [], "labels": [], "codes": []}

    for i, (sentence, mask, code) in enumerate(zip(sentences, masks, codes)):

        # Get span limits through its mask
        span_seps = [pos for pos, char in enumerate(mask.strip()) if char == ' ']

        spans = []
        cursor = -1
        # Break sentence with multiple spans into a list of its respective spans.
        for sep in span_seps:
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
            # Edge case scenario (should not occur)
            return {"sentence_id": [], "words": [], "labels": [], "codes": []}

        result["sentence_id"].extend([i] * len(tokenized_spans))
        result["words"].extend(tokenized_spans)
        result["labels"].extend(tokenized_masks)
        result["codes"].extend(tokenized_codes)

    return result


def fix_sentence_id(dfs):
    """
    Makes sentence_id unique by taking into consideration the number of sentences already seen in previous dataframes
    :param dfs: List[DataFrame]
    :return: List[DataFrame]
    """
    offset = 0
    for df in dfs:

        # Include offset
        df.sentence_id += offset

        # Number of total sentences from previous dataframes in list
        offset += df.shape[0]

    return dfs


def split_dfs(dfs, train_perc, test_perc):
    """
    Split Dataframes according to a certain Train/Test/Val proportion. This value might not be exact.
    :param dfs: List[DataFrames] to be split into 3 DataFrames according to train_perc and test_perc
    :param train_perc: Train dataframe percentage
    :param test_perc: Test dataframe percentage (considers only Test and not Val)
    :return: 3 DataFrames Train/Test/Val
    """
    # Lengths of each dataframe
    lens = {i: df.shape[0] for i, df in enumerate(dfs)}

    total = sum(lens.values())

    train_size = total * train_perc
    test_size = total * (train_perc + test_perc)

    cursor = 0
    seps = []
    for dfi, size in lens.items():
        cursor += size
        # Greedily grab the Train set
        if cursor > train_size and not seps:
            seps.append(dfi)
        # Leave rest for validation set
        elif cursor > test_size and len(seps) == 1:
            seps.append(dfi)

    return pd.concat(dfs[:seps[0]+1]), pd.concat(dfs[seps[0]+1:seps[1]+1]), pd.concat(dfs[seps[1]+1:])


# String with all letters (both capitalized and lower case) and numbers, to be used as unary category representation
alphanum = "{}{}".format(digits, ascii_letters)

# Code to Unit
u_code = {str(code): alphanum[ic] for ic, code in enumerate(pd.concat(d[0] for d in df_list).Code.unique())}

# Unit to Code
inv_u_code = {v: k for k, v in u_code.items()}

tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased')

token_dfs = []

# Loop through all manifestos
for manifesto_df, country in tqdm(df_list, desc="Processing DataFrames: "):

    manifesto_text = ""
    masked_text = ""
    masked_code = ""

    for index, row in manifesto_df.iterrows():
        # Clean up a little
        text = " ".join(sent_tokenize(" ".join(row.Text.split())))

        # Remove special cases found
        text = text.replace('\uf020', '').replace('//', '--').replace(';.', '; ')

        # BIO mask
        code = str(row.Code)
        bio = "B" if code not in ('000', '0', '999') else 'O'
        mask_in = "I" if bio == 'B' else 'O'
        text_mask = bio + mask_in * (len(text) - 1)

        # Code mask
        code_mask = u_code[code] * len(text)

        # Everything is a continuous string
        masked_text += " {}".format(text_mask)
        manifesto_text += " {}".format(text)
        masked_code += " {}".format(code_mask)

    # More clean up
    manifesto_text = manifesto_text.strip()
    masked_text = masked_text.strip()
    masked_code = masked_code.strip()

    # capitalize first letter after semi-colon (important for span separation at semi-colon)
    manifesto_text = re.sub('(; [a-z0-9])', capitalize, manifesto_text)

    # Use semi-colon as a span delimiter
    manifesto_text = manifesto_text.replace(';', '.')
    sent_list = sent_tokenize(manifesto_text)

    # Match masks with new sentences
    mask_list = match_mask(sent_list, masked_text)
    code_list = match_mask(sent_list, masked_code)

    # Process items
    sent_list, mask_list, code_list = sentence_fix(sent_list, mask_list, code_list)
    sent_list, mask_list, code_list = chunk_sentences(tokenizer, sent_list, mask_list, code_list)
    tokens = tokenize_sentence(sent_list, mask_list, code_list)

    # Build dataframe for manifesto
    data = pd.DataFrame(tokens)

    # Return unary code into MAN category
    data.codes = data.codes.apply(lambda x: inv_u_code[x])

    # Add information about country of origin
    data["country"] = country

    token_dfs.append(data)

token_dfs = fix_sentence_id(token_dfs)

train, test, val = split_dfs(token_dfs, 0.7, 0.15)

print("Train:{}\nTest:{}\nVal:{}".format(train.shape[0], test.shape[0], val.shape[0]))

train.to_csv(TRAIN_DIR + "train_ner.csv", index=False)
val.to_csv(VAL_DIR + 'val_ner.csv', index=False)
test.to_csv(TEST_DIR + 'test_ner.csv', index=False)
