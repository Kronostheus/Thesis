import openpyxl as px
import glob

WORKBOOK_DIR = "Data/Workbooks/"

COLS = ['C', 'F']

STYLE_VALUES = {"Good": 1, "Neutral": 0.5, "Bad": 0}
VALUES_STYLE = {1: "Good", 0.5: "Neutral", 0: "Bad"}

files = [file for file in glob.glob(WORKBOOK_DIR + '*.xlsx')]

to_process = [file for file in files if px.load_workbook(file).active['C2'].style not in STYLE_VALUES.keys()]
to_match = [file for file in files if file not in to_process]


def decide_style(topic_dict, cell):
    best_style = VALUES_STYLE[max(STYLE_VALUES[cell.style], STYLE_VALUES[topic_dict[cell.internal_value]])] \
        if cell.internal_value in topic_dict.keys() else cell.style
    return best_style


last_wb_dict = {}

for file in to_match:
    last_wb = px.load_workbook(file).active
    for row in range(2, last_wb.max_row + 1):
        first_col, second_col = 'C' + str(row), 'F' + str(row)
        first_topic, second_topic = last_wb[first_col], last_wb[second_col]

        if row in last_wb_dict.keys():
            last_wb_dict[row][first_topic.internal_value] = decide_style(last_wb_dict[row], first_topic)
            last_wb_dict[row][second_topic.internal_value] = decide_style(last_wb_dict[row], second_topic)
        else:
            last_wb_dict[row] = {first_topic.internal_value: first_topic.style,
                                 second_topic.internal_value: second_topic.style}

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
