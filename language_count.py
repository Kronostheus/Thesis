import glob
import pandas as pd

country_dict = {
    "pt": "Data/Portugal_Manifestos/",
    "br": "Data/Brazil_Manifestos/",
    "sp": "Data/Spain_Media/",
    "it": "Data/Italy_Manifestos/"}

df = None

for lang, path in country_dict.items():
    lang_df = pd.concat([pd.read_csv(path) for path in glob.glob(path + '*.csv')])
    count_df = lang_df.Code.value_counts().to_frame().reset_index()
    count_df.columns = ["Code", "{}".format(lang.upper())]

    if df is None:
        df = count_df
    else:
        df = pd.merge(df, count_df, on='Code', how='left')

df.fillna(0, inplace=True)

split_df = pd.read_csv('Data/Classification/dataset_split.csv')
split_df = split_df[['Code', 'Main_Count', 'Main_Perc']]

new_df = pd.merge(df, split_df, on='Code', how='left')
new_df["Total"] = new_df.apply(lambda x: "{} ({})".format(int(x.Main_Count), round(x.Main_Perc, 4)), axis=1)
new_df.sort_values(by='Main_Count', ascending=False, inplace=True)

names = pd.read_csv('Data/Coding_Schemes/final_man.csv').dropna().reset_index(drop=True)

names = pd.merge(names, new_df, left_on='MAN Code', right_on='Code', how='left').drop(columns=['Code', 'Main_Count', 'Main_Perc'])
names[["MAN Code", "PT", "BR", "IT", "SP"]] = names[["MAN Code", "PT", "BR", "IT", "SP"]].astype(int)
names = names.astype(str).to_csv('Data/lang_count.csv', index=False)
breakpoint()
