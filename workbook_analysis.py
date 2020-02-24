import openpyxl as px
import pandas as pd
import glob

WORKBOOK_DIR = "Data/Workbooks/"

COLS = ['C', 'F']
STYLE_VALUE = {'Bad': 0, 'Neutral': 0.5, 'Good': 1}

index_colors = px.styles.colors.COLOR_INDEX

files = [file for file in glob.glob(WORKBOOK_DIR + '*.xlsx')]


def build_dataframe():
    col_dict = {}

    for file in files:
        idx = str(files.index(file) + 1)
        sheet = px.load_workbook(file).active

        for col in COLS:
            col_dict[idx + col] = [STYLE_VALUE[sheet[col + str(cell_num)].style] for cell_num in range(2, sheet.max_row + 1)]

    return pd.DataFrame(col_dict)


sheet_df = build_dataframe()
file_cols = [list(sheet_df.columns)[start:start+2] for start in range(0, sheet_df.shape[1] - 1, 2)]


def print_out():
    for file_index in range(len(files)):
        cols = file_cols[file_index]
        points_count = list(dict(sheet_df[cols].sum()).values())
        value_counts = [dict(sheet_df[col].value_counts()) for col in cols]

        good_sec = sheet_df[cols][(sheet_df[cols[0]] != 1) & (sheet_df[cols[1]] == 1)].shape[0]

        print(files[file_index])
        print("Points:\n\tFirst Choice: {}\n\tSecond Choice: {}".format(points_count[0], points_count[1]))
        print("Counts:\n"
              "\tFirst Choice:\n"
              "\t\tGood: {}\n"
              "\t\tNeutral: {}\n"
              "\t\tBad: {}\n"
              "\tSecond Choice:\n"
              "\t\tGood: {}\n"
              "\t\tNeutral: {}\n"
              "\t\tBad: {}\n".format(value_counts[0].get(1.0), value_counts[0].get(0.5), value_counts[0].get(0),
                                     value_counts[1].get(1.0), value_counts[1].get(0.5), value_counts[1].get(0))
              )
        print("A total of {} Good categories found in second choice, while first either Bad or Neutral.\n"
              "If we were to replace these, the original {} Good choices would total {} out of {}.\n"
              "That would represent {}% of Good choices.\n".format(good_sec, value_counts[0].get(1.0),
                                                                   value_counts[0].get(1.0) + good_sec,
                                                                   sheet_df.shape[0],
                                                                   round(100 * (value_counts[0].get(1.0) + good_sec)
                                                                         / sheet_df.shape[0]), 2)
              )
print_out()

print()