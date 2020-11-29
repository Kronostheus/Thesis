import pandas as pd
from googletrans import Translator
from tqdm import tqdm
import datetime
from concurrent.futures import ThreadPoolExecutor
from Utils import Config

# Each language is translated into all the others
dests = {val: list(set([dst for dst in Config.langs.values() if dst != val])) for val in Config.langs.values()}

# train = pd.read_csv(Config.TRAIN_DIR + 'train.csv', header=0)
# train['source'] = train.country.apply(lambda country: Config.langs[country])

# batch_size = 10000
# dfs = [train[i:i+batch_size] for i in range(0, train.shape[0], batch_size)]


def process_row(row):
    """
    Given a original span, translated it into all other languages and return a list with all possibilities.
    Will return original span as first element.
    All returned spans have their Text and Code include in order to be easily converted into a dataframe row.
    :param row: np.Series representing a row from a dataframe
    :return: List with future rows
    """
    # This originally was a one liner but API is not exactly very stable
    # dest_langs = dests[row.source]
    # org = (row.Text, row.Code)
    # t1 = (translator.translate(row.Text, src=row.source, dest=dest_langs[0]).text, row.Code)
    # t2 = (translator.translate(row.Text, src=row.source, dest=dest_langs[1]).text, row.Code)

    response = {(translator.translate(row.Text, src=row.source, dest=dest).text, row.Code) for dest in dests[row.source]}
    response.update({(row.Text, row.Code)})
    return list(response)


# # Select which dataframe to process
# df = dfs[14]
row_lst = []
train = list(pd.read_csv(Config.DATA_DIR + 'train.csv', header=0, chunksize=500))
prev_backup = pd.read_csv(Config.DATA_DIR + 'backup.csv', header=0)

df_idx = 0
translator = Translator()
print(datetime.datetime.now())


def translate_batches(idx):
    print("Starting translation on batch #{}".format(idx))
    global translator
    global row_lst
    # Do not overwhelm the API
    for i, d in tqdm(enumerate(train[idx:]), total=len(train[idx:])):

        d['source'] = d.country.apply(lambda country: Config.langs[country])

        # Get new Translator object
        translator = Translator()

        try:
            # Distribute processing to avoid 2h/dataframe
            with ThreadPoolExecutor(max_workers=20) as executor:
                row_lst.extend(
                    executor.map(process_row, d.itertuples(index=False))
                )
        except Exception as e:
            print("Restarting translations: {}".format(e))
            global df_idx
            df_idx += i
            translate_batches(df_idx)

        backup_rows = [r for trsl in row_lst for r in trsl]

        backup = pd.DataFrame(backup_rows, columns=['Text', 'Code'])

        pd.concat([prev_backup, backup]).to_csv(Config.DATA_DIR + 'backup.csv', index=False)

        del translator


translate_batches(df_idx)

print(datetime.datetime.now())

row_lst = [r for trsl in row_lst for r in trsl]

pd.DataFrame(row_lst, columns=['Text', 'Code']).sample(frac=1, random_state=42).to_csv(Config.DATA_DIR + 'train_trans.csv', index=False)

