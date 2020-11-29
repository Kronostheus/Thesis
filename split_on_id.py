import pandas as pd
import re
from pathlib import Path
from Utils.config import Config
from Utils.utils import build_span_df
from sklearn.model_selection import train_test_split


def get_all_freqs():
    SPAIN_DATA = Path(Config.DATA_DIR + '/Spain_Media/').glob('*.csv')
    PORTUGAL_DATA = Path(Config.DATA_DIR + '/Portugal_Manifestos/').glob('*.csv')
    BRAZIL_DATA = Path(Config.DATA_DIR + '/Brazil_Manifestos/').glob('*.csv')
    ITALY_DATA = Path(Config.DATA_DIR + '/Italy_Manifestos/').glob('*.csv')

    all_data = pd.DataFrame()

    for country, data_generator in zip(["P", "S", "B", "I"], [PORTUGAL_DATA, SPAIN_DATA, BRAZIL_DATA, ITALY_DATA]):
        country_df = pd.concat([pd.read_csv(path) for path in data_generator])
        country_df["country"] = country
        all_data = pd.concat([all_data, country_df])

    return all_data.Code.value_counts(normalize=True)


train_ner = pd.read_csv(Config.DATA_DIR + 'train_ner_spans.csv')
val_ner = pd.read_csv(Config.DATA_DIR + 'val_ner_spans.csv')
test_ner = pd.read_csv(Config.DATA_DIR + 'test_ner_spans.csv')

train_ = pd.read_csv(Config.TRAIN_DIR + 'train.csv')
train_spain = train_[train_.country == 'S']
train = pd.concat([train_ner, train_spain])
t = pd.DataFrame([train_.Code.value_counts(normalize=True), train.Code.value_counts(normalize=True)])

test_ = pd.read_csv(Config.TEST_DIR + 'test.csv')
test_spain = test_[test_.country == 'S']
test = pd.concat([test_ner, test_spain])
t_ = pd.DataFrame([test_.Code.value_counts(normalize=True), test.Code.value_counts(normalize=True)])

val_ = pd.read_csv(Config.VAL_DIR + 'val.csv')
val_spain = val_[val_.country == 'S']
val = pd.concat([val_ner, val_spain])
v = pd.DataFrame([val_.Code.value_counts(normalize=True), val.Code.value_counts(normalize=True)])


def untokenize(text):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .', '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ").replace("( ", " (").replace(" )", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" ` ", " '")
    return step5.strip()


def preprocess_text(text):
    text = re.sub(u"([\u2018\u2019])", "'", text)
    text = re.sub('([«»])', '"', text)
    text = re.sub('\"\"+', '\"', text)
    text = re.sub('--*', '-', text)
    text = " ".join(text.split())
    return untokenize(text)


test.Text = test.Text.apply(lambda x: preprocess_text(str(x)))
val.Text = val.Text.apply(lambda x: preprocess_text(str(x)))
train.Text = train.Text.apply(lambda x: preprocess_text(str(x)))

# test.sample(frac=1, random_state=42).reset_index(drop=True).to_csv(Config.DATA_DIR + 'test.csv', index=False)
# train.sample(frac=1, random_state=42).reset_index(drop=True).to_csv(Config.DATA_DIR + 'train.csv', index=False)
# val.sample(frac=1, random_state=42).reset_index(drop=True).to_csv(Config.DATA_DIR + 'val.csv', index=False)
