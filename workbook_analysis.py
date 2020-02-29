import openpyxl as px
import pandas as pd
import glob

WORKBOOK_DIR = "Data/Workbooks/"

COLS = ['C', 'F', 'I', 'L', 'O']
STYLE_VALUE = {'Bad': 0, 'Neutral': 0.5, 'Good': 1}

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
file_cols = [list(sheet_df.columns)[start:start+5] for start in range(0, sheet_df.shape[1] - 1, 5)]


def print_out():
    for file_index in range(len(files)):
        cols = file_cols[file_index]

        good_first = sheet_df[cols][sheet_df[cols[0]] == 1].shape[0]

        good_sec = sheet_df[cols][(sheet_df[cols[0]] != 1) &
                                  (sheet_df[cols[1]] == 1)].shape[0]

        good_third = sheet_df[cols][(sheet_df[cols[0]] != 1) &
                                    (sheet_df[cols[1]] != 1) &
                                    (sheet_df[cols[2]] == 1)].shape[0]

        good_fourth = sheet_df[cols][(sheet_df[cols[0]] != 1) &
                                     (sheet_df[cols[1]] != 1) &
                                     (sheet_df[cols[2]] != 1) &
                                     (sheet_df[cols[3]] == 1)].shape[0]

        good_fifth = sheet_df[cols][(sheet_df[cols[0]] != 1) &
                                     (sheet_df[cols[1]] != 1) &
                                     (sheet_df[cols[2]] != 1) &
                                     (sheet_df[cols[3]] != 1) &
                                     (sheet_df[cols[4]] == 1)].shape[0]

        good_neither = sheet_df[cols][(sheet_df[cols[0]] != 1) &
                                     (sheet_df[cols[1]] != 1) &
                                     (sheet_df[cols[2]] != 1) &
                                     (sheet_df[cols[3]] != 1) &
                                     (sheet_df[cols[4]] != 1)].shape[0]

        print("##################" + files[file_index] + "##################")

        print("A total of {} Good categories found in second choice, while first either Bad or Neutral.\n"
              "If we were to replace these, the original {} Good choices would total {} out of {}.\n"
              "That would represent {}% of Good choices.\n".format(good_sec,
                                                                   good_first,
                                                                   good_first + good_sec,
                                                                   sheet_df.shape[0],
                                                                   round(100 * (good_first + good_sec)
                                                                         / sheet_df.shape[0], 2))
              )

        print("A total of {} Good categories found in third choice, while first either Bad or Neutral.\n"
              "If we were to replace these, the previous {} Good choices would total {} out of {}.\n"
              "That would represent {}% of Good choices.\n".format(good_third,
                                                                   good_first + good_sec,
                                                                   good_first + good_sec + good_third,
                                                                   sheet_df.shape[0],
                                                                   round(100 * (good_first + good_sec + good_third)
                                                                         / sheet_df.shape[0], 2))
              )

        print("A total of {} Good categories found in fourth choice, while first either Bad or Neutral.\n"
              "If we were to replace these, the previous {} Good choices would total {} out of {}.\n"
              "That would represent {}% of Good choices.\n".format(good_fourth,
                                                                   good_first + good_sec + good_third,
                                                                   good_first + good_sec + good_third + good_fourth,
                                                                   sheet_df.shape[0],
                                                                   round(100 * (good_first + good_sec + good_third + good_fourth)
                                                                         / sheet_df.shape[0], 2))
              )

        print("A total of {} Good categories found in fifth choice, while first either Bad or Neutral.\n"
              "If we were to replace these, the previous {} Good choices would total {} out of {}.\n"
              "That would represent {}% of Good choices.\n".format(good_third,
                                                                   good_first + good_sec + good_third + good_fourth,
                                                                   good_first + good_sec + good_third + good_fourth + good_fifth,
                                                                   sheet_df.shape[0],
                                                                   round(100 * (good_first + good_sec + good_third + good_fourth + good_fifth)
                                                                         / sheet_df.shape[0], 2))
              )

        print("There are a total of {} topics with no Good matches ({}%)\n".format(good_neither, round(100 * good_neither / sheet_df.shape[0], 2)))

print_out()

cap_code_df = pd.read_csv("Data/Coding_Schemes/Matchings/best_matches5.csv").CAP
no_good = sheet_df[file_cols[-1]][(sheet_df[file_cols[-1][0]] != 1) &
                                     (sheet_df[file_cols[-1][1]] != 1) &
                                     (sheet_df[file_cols[-1][2]] != 1) &
                                     (sheet_df[file_cols[-1][3]] != 1) &
                                     (sheet_df[file_cols[-1][4]] != 1)]

df = pd.concat([cap_code_df, no_good], axis=1).dropna()
print()