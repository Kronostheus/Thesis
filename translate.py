import pandas as pd
from googletrans import Translator
from tqdm import tqdm
import datetime
from concurrent.futures import ThreadPoolExecutor

# API languages
langs = {'P': 'pt', 'S': 'es', 'I': 'it', 'B': 'pt'}

# Each language is translated into all the others
dests = {val: list(set([dst for dst in langs.values() if dst != val])) for val in langs.values()}

train = pd.read_csv('Data/Classification/Train/train.csv', header=0)
train['source'] = train.country.apply(lambda country: langs[country])

batch_size = 10000
dfs = [train[i:i+batch_size] for i in range(0, train.shape[0], batch_size)]

# final = pd.read_csv('Data/Classification/Train/translate_trans.csv')
# final = final[~final.duplicated()]
# final.to_csv('Data/Classification/Train/train_translated.csv', index=0)


def process_row(row):
    # This originally was a one liner but API is not exactly very stable
    dest_langs = dests[row.source]
    org = [row.Text, row.Code]
    t1 = [translator.translate(row.Text, src=row.source, dest=dest_langs[0]).text, row.Code]
    t2 = [translator.translate(row.Text, src=row.source, dest=dest_langs[1]).text, row.Code]
    return [org, t1, t2]


# Select which dataframe to process
df = dfs[14]
row_lst = []

translator = Translator()

print(datetime.datetime.now())

# Do not overwhelm the API
for d in tqdm([df[i:i+500] for i in range(0, df.shape[0], 500)]):

    # Distribute processing to avoid 2h/dataframe
    with ThreadPoolExecutor(max_workers=20) as executor:
        row_lst.extend(
            executor.map(process_row, d.itertuples(index=False))
        )

    # Get new Translator object
    translator = Translator()

print(datetime.datetime.now())

row_lst = [r for trsl in row_lst for r in trsl]

data = pd.DataFrame(row_lst, columns=['Text', 'Code'])
print(data[data.duplicated()].shape[0])
data.to_csv('Data/Classification/Train/t.csv', index=False)

backup = pd.read_csv('Data/Classification/Train/translate_trans.csv')
backup.to_csv('Data/Classification/Train/backup.csv', index=False)

cdata = pd.concat([pd.read_csv('Data/Classification/Train/translate_trans.csv'), data], ignore_index=True)
cdata.to_csv('Data/Classification/Train/translate_trans.csv', index=False)
