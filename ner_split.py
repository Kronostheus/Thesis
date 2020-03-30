import pandas as pd
import glob
from string import punctuation
from random import shuffle
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

CLASS_DIR = "Data/Classification/"
DATA_DIR = "Data/"

TRAIN_DIR = CLASS_DIR + 'Train/'
VAL_DIR = CLASS_DIR + 'Val/'
TEST_DIR = CLASS_DIR + 'Test/'

PORTUGAL_DATA = glob.glob(DATA_DIR + '/Portugal_Manifestos/*.csv')
BRAZIL_DATA = glob.glob(DATA_DIR + '/Brazil_Manifestos/*.csv')

all_data = PORTUGAL_DATA + BRAZIL_DATA
shuffle(all_data)

data = pd.DataFrame()
sizes = [0]
for path in all_data:
    df = pd.read_csv(path)
    sizes.append(sizes[-1] + df.shape[0])
    data = pd.concat([data, df])
    #data = pd.concat([data, pd.concat([pd.read_csv(path) for path in data_generator])])


def label_span(row):
    span_lst = word_tokenize(row.Text)
    cat = str(row.Code)
    lst = []
    if cat == '000' or cat == '999':
        lst = [[e, "O"] for e in span_lst]
    else:
        for i in range(len(span_lst)):
            bio = "B" if i == 0 else "I"
            element = span_lst[i]
            # if all(char in punctuation for char in element):
            #     bio += "-PUNCT"
            # else:
            #     bio += "-SPAN"
            lst.append([element, bio])

    return lst


def build_list(df):
    return [lst for lst in df.apply(lambda x: label_span(x), axis=1)]


def build_dataframe(lst):
    return pd.DataFrame([item for sublist in lst for item in sublist], columns=["token", "bio"])


def pipeline(df):
    return build_dataframe(build_list(df))


def get_b(df, perc):
    i = int(perc * df.shape[0])
    j = i - 1
    while True:

        if i > df.shape[0] or j < 0:
            raise Exception("Out of Bounds")

        if df.iat[i, 1] == 'B':
            return i
        elif df.iat[j, 1] == 'B':
            return j

        i += 1
        j -= 1


all_ner = pipeline(data)
train_test = get_b(all_ner, 0.7)
train, test_ = all_ner.iloc[:train_test], all_ner.iloc[train_test:]
test_val = get_b(test_, 0.5)
test, val = test_.iloc[:test_val], test_.iloc[test_val:]

train.to_csv(TRAIN_DIR + "train_ner.csv", index=False)
val.to_csv(VAL_DIR + 'val_ner.csv', index=False)
test.to_csv(TEST_DIR + 'test_ner.csv', index=False)
