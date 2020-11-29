import pandas as pd
from pathlib import Path
from Utils import Config
from tqdm import tqdm

data = [
    ('train', pd.read_csv(Config.TRAIN_DIR + 'train.csv')),
    ('train_translated', pd.read_csv(Config.TRAIN_DIR + 'train_translated.csv')),
    ('train_combinations', pd.read_csv(Config.TRAIN_DIR + 'train_combinations.csv')),
    ('test', pd.read_csv(Config.TEST_DIR + 'test.csv')),
    ('val', pd.read_csv(Config.VAL_DIR + 'val.csv'))
]

Path('Data/LanguageModeling/').mkdir(exist_ok=True)

for filename, df in data:
    print("Processing: {}".format(filename))

    df = df[~df.duplicated()]
    text = df.Text.astype(str).values

    # Important: ensure UTF-8 encoding
    with open('Data/LanguageModeling/{}.txt'.format(filename), 'w', encoding='utf-8') as file:
        # Each span is a line in the text file
        for span in tqdm(text):
            try:
                file.write('{}\n'.format(span))
            except Exception as e:
                print("Encountered Error: {}\nText: {}".format(e, span))
                # Ignore potential errors
                continue

