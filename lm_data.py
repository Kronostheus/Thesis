import pandas as pd
from pathlib import Path

# data = [('train', pd.read_csv('Data/Classification/Train/train.csv')),
#         ('test', pd.read_csv('Data/Classification/Test/test.csv')),
#         ('val', pd.read_csv('Data/Classification/Val/val.csv'))]

data = [('train_translated', pd.read_csv('Data/Classification/Train/train_translated.csv').sample(frac=1, random_state=42))]

Path('Data/LanguageModeling/').mkdir(exist_ok=True)

for filename, df in data:
    text = df.Text.astype(str).values
    # Important: ensure UTF-8 encoding
    with open('Data/LanguageModeling/{}.txt'.format(filename), 'w', encoding='utf-8') as file:
        # Each span is a line in the text file
        for span in text:
            try:
                file.write('{}\n'.format(span))
            except:
                # Ignore potential errors
                continue

