import glob
import pandas as pd
import numpy as np
from functools import reduce
from Utils.config import Config

langs = {'P': 'PT', 'B': 'BR', 'I': 'IT', 'S': 'SP'}
csv_files = [Config.TRAIN_DIR + 'train.csv', Config.TEST_DIR + 'test.csv', Config.VAL_DIR + 'val.csv']

df = pd.concat([pd.read_csv(file) for file in csv_files])


def make_string(row):
    return "{} ({}%)".format(int(row.Count), round(row.Perc * 100, 2))


def count_words(code):
    return int(np.mean([len(text.split()) for text in df[df.Code == code].Text]))


total = df.Code.value_counts().to_frame().reset_index()
total.columns = ['Code', 'Count']
total['Perc'] = df.Code.value_counts(normalize=True).to_list()


code_dfs = []
for country, country_df in df.groupby(by='country'):
    country_cats = country_df.Code.value_counts().to_frame().reset_index()
    country_cats.columns = ['Code', langs[country]]
    code_dfs.append(country_cats)

code_dfs.append(total)
joined_df = reduce(lambda x, y: pd.merge(x, y, on='Code', how='outer'), code_dfs)
joined_df.fillna(0, inplace=True)
joined_df['Total (%)'] = joined_df.apply(make_string, axis=1)
joined_df.sort_values(by='Count', ascending=False, inplace=True)
joined_df.drop(columns=['Count', 'Perc'], inplace=True)

names = pd.read_csv('Data/Coding_Schemes/final_man.csv').dropna().reset_index(drop=True)
names.columns = ['Topic', 'MAN Code']
names = pd.merge(names, joined_df, left_on='MAN Code', right_on='Code', how='left').drop(columns=['Code'])
names[["MAN Code", "PT", "BR", "IT", "SP"]] = names[["MAN Code", "PT", "BR", "IT", "SP"]].astype(int)
names['Avg. Words'] = names['MAN Code'].apply(count_words)
names = names[['Topic', 'MAN Code', 'Avg. Words', 'PT', 'BR', 'IT', 'SP', 'Total (%)']].astype(str)
# names.to_csv('Data/lang_count.csv', index=False)
breakpoint()
