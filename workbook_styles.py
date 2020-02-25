import openpyxl as px
import pandas as pd
import glob

WORKBOOK_DIR = "Data/Workbooks/"

COLS = ['C', 'F']
STYLE_VALUE = {0: 'Bad', 0.5: 'Neutral', 1: 'Good'}

files = [file for file in glob.glob(WORKBOOK_DIR + '*.xlsx')]

to_process = [file for file in files if px.load_workbook(file).active['C2'].style not in STYLE_VALUE.values()]
to_match = [file for file in files if file not in to_process][-1]

last_wb = px.load_workbook(to_match).active

last_wb_dict = {}

for row in range(2, last_wb.max_row + 1):
    first_topic, second_topic = 'C' + str(row), 'F' + str(row)
    last_wb_dict[row] = {last_wb[first_topic].internal_value: last_wb[first_topic].style,
                         last_wb[second_topic].internal_value: last_wb[second_topic].style}

for file in to_process:
    wb = px.load_workbook(file)
    sheet = wb.active
    for row in range(2, sheet.max_row + 1):
        prev_values = last_wb_dict[row]
        first_topic, second_topic = 'C' + str(row), 'F' + str(row)

        sheet[first_topic].style = prev_values[sheet[first_topic].internal_value] \
            if sheet[first_topic].internal_value in prev_values.keys() else "Normal"

        sheet[second_topic].style = prev_values[sheet[second_topic].internal_value] \
            if sheet[second_topic].internal_value in prev_values.keys() else "Normal"

    wb.save(file)
