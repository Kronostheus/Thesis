import openpyxl as px
import glob

WORKBOOK_DIR = "Data/Workbooks/"

COLS = ['C', 'F']

STYLE_VALUES = {"Good": 1, "Neutral": 0.5, "Bad": 0, "Normal": -1}
VALUES_STYLE = {1: "Good", 0.5: "Neutral", 0: "Bad", -1: "Normal"}

files = [file for file in glob.glob(WORKBOOK_DIR + '*.xlsx')]

to_process = [file for file in files if px.load_workbook(file).active['O2'].style == "Normal"]
to_match = [file for file in files if file not in to_process]


def decide_style(topic_dict, cell):
    best_style = VALUES_STYLE[max(STYLE_VALUES[cell.style], STYLE_VALUES[topic_dict[cell.internal_value]])] \
        if cell.internal_value in topic_dict.keys() else cell.style
    return best_style


def get_cols(row):
    return [col + str(row) for col in ['C', 'F', 'I', 'L', 'O']]

last_wb_dict = {}

for file in files:
    last_wb = px.load_workbook(file).active
    for row in range(2, last_wb.max_row + 1):
        #first_col, second_col, third_col, fourth_col, fifth_col = get_cols(row)
        first_topic, second_topic, third_topic, fourth_topic, fifth_topic = [last_wb[col] for col in get_cols(row)]

        if row in last_wb_dict.keys():
            last_wb_dict[row][first_topic.internal_value] = decide_style(last_wb_dict[row], first_topic)
            last_wb_dict[row][second_topic.internal_value] = decide_style(last_wb_dict[row], second_topic)
            last_wb_dict[row][third_topic.internal_value] = decide_style(last_wb_dict[row], third_topic)
            last_wb_dict[row][fourth_topic.internal_value] = decide_style(last_wb_dict[row], fourth_topic)
            last_wb_dict[row][fifth_topic.internal_value] = decide_style(last_wb_dict[row], fifth_topic)
        else:
            last_wb_dict[row] = {first_topic.internal_value: first_topic.style,
                                 second_topic.internal_value: second_topic.style,
                                 third_topic.internal_value: third_topic.style,
                                 fourth_topic.internal_value: fourth_topic.style,
                                 fifth_topic.internal_value: fifth_topic.style}

for file in to_process:
    wb = px.load_workbook(file)
    sheet = wb.active
    for row in range(2, sheet.max_row + 1):
        prev_values = last_wb_dict[row]
        first_topic, second_topic, third_topic, fourth_topic, fifth_topic = get_cols(row)

        sheet[first_topic].style = prev_values[sheet[first_topic].internal_value] \
            if sheet[first_topic].internal_value in prev_values.keys() else sheet[first_topic].style

        sheet[second_topic].style = prev_values[sheet[second_topic].internal_value] \
            if sheet[second_topic].internal_value in prev_values.keys() else sheet[second_topic].style

        sheet[third_topic].style = prev_values[sheet[third_topic].internal_value] \
            if sheet[third_topic].internal_value in prev_values.keys() else sheet[third_topic].style

        sheet[fourth_topic].style = prev_values[sheet[fourth_topic].internal_value] \
            if sheet[fourth_topic].internal_value in prev_values.keys() else sheet[fourth_topic].style

        sheet[fifth_topic].style = prev_values[sheet[fifth_topic].internal_value] \
            if sheet[fifth_topic].internal_value in prev_values.keys() else sheet[fifth_topic].style

    wb.save(file)
