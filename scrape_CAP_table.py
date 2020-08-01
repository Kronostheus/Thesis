import pandas as pd
from Utils import Config

print_tables = False


def build_table(df):
    rows = []
    for major, major_df in df.groupby(by='Major Topic', sort=False):
        rows.append([
            major,
            "; ".join(major_df.apply(lambda x: "{} ({})".format(x['Minor Topic'], str(x['Code'])), axis=1).to_list())
        ])
    return pd.DataFrame(rows, columns=['Major Topic', 'Minor Topics'])


cap = pd.read_csv(Config.SCHEMES_DIR + 'CAP.csv')
man = pd.read_csv(Config.SCHEMES_DIR + 'MAN_v4.csv')

cap_table = build_table(cap)
man_table = build_table(man)

if print_tables:
    cap_table.to_csv(Config.SCHEMES_DIR + 'CAP_table.csv', index=False)
    man_table.to_csv(Config.SCHEMES_DIR + 'MAN_table.csv', index=False)
