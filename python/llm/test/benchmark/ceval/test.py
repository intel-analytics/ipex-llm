import os
import pdb
import numpy as np
import pandas as pd
from pathlib import Path

def highlight_vals(val, max=3.0, color1='red', color2='green', color3='yellow', is_last=False):
    if isinstance(val, float):
        if val > max:
            return 'background-color: %s' % color2
        elif val <= -max:
            return 'background-color: %s' % color1
        elif val != 0.0 and is_last:
            return 'background-color: %s' % color3
    else:
        return ''

def calculate_diff(cur_array, previous_array):
    new_array = []
    for i in range(len(cur_array)):
        if type(cur_array[i]) == type(pd.NA) or type(previous_array[i]) == type(pd.NA):
            new_array.append(pd.NA)
        else:
            new_array.append(round((cur_array[i]-previous_array[i])*100/previous_array[i], 2))
    return np.array(new_array)

def calculate_percentage_difference(cur_array, previous_array):
    new_array = []
    for i in range(len(cur_array)):
        if type(cur_array[i]) == type(pd.NA) or type(previous_array[i]) == type(pd.NA):
            new_array.append(pd.NA)
        else:
            new_array.append(round((cur_array[i]-previous_array[i])*100/previous_array[i], 2))
    return np.array(new_array)

def check_diffs_within_normal_range(latest_csv, highlight_set, threshold):
    within = True

    for column in highlight_set:
        for value in latest_csv[column]:
            if type(value) != type(pd.NA):
                within = within and abs(value) <= threshold
    
    return within


if __name__ == '__main__':
    previous_info = {'Model Name': [],
                    'Precision': [],
                    'STEM': [],
                    'Social Science': [],
                    'Humanities': [],
                    'Other': [],
                    'Hard': [],
                    'Average': []}
    # data = pd.read_csv('fp16.csv', index_col=0)

    latest_csv = pd.read_csv('latest.csv', index_col=0)
    previous_csv = pd.read_csv('previous.csv', index_col=0)
    previous_csv['Average'][1] = pd.NA

    subjects = ['STEM', 'Social Science', 'Humanities', 'Other', 'Hard', 'Average']
    precisions = ['sym_int4', 'fp8_e5m2']
    highlight_set = []

    insert_column = latest_csv.shape[-1]
    # in the make_csv step we will handle the missing values and make it pd.NA
    for subject in subjects:
        # insert last accuracy task
        latest_csv.insert(loc=insert_column, column=f'last_{subject}',
                            value=previous_csv[subject])
        # np.where(previous_csv[subject]<=0.0, pd.NA, previous_csv[subject])

        # insert precentage difference between previous and current value
        latest_csv.insert(
            loc=insert_column+1,
            column=f'diff_{subject}(%)',
            value=calculate_percentage_difference(latest_csv[subject], previous_csv[subject]))
        # append in the highlight set
        highlight_set.append(f'diff_{subject}(%)')

        # update insert column
        insert_column += 2

    columns = {}
    for column in latest_csv.columns.values.tolist():
        columns[column] = '{:.2f}'

    check_diffs_within_normal_range(latest_csv, highlight_set, 3.0)

    pdb.set_trace()
    styled_df = latest_csv.style.format(columns).applymap(lambda val: highlight_vals(val, max=3.0, is_last=True), subset=highlight_set)



