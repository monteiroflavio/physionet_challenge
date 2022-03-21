import os
import re
import decimal

decimal.getcontext().prec = 3

# testing whether all expected patterns really exists
def testing_expected_patterns (set, set_path):
    patterns = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']
    failed_patterns = {
        'RecordID' : [],
        'Age' : [],
        'Gender' : [],
        'Height' : [],
        'ICUType' : [],
        'Weight' : []
    }
    flag = False

    for pattern in patterns:
        for file in set:
            for line in open(os.path.join(set_path, file), 'r').read().split('\n'):
                if re.match(r'(0?0:0?0),('+pattern+')', line):
                    flag = True
            if flag == False:
                failed_patterns[pattern].append(file)
            flag = False

    return failed_patterns

# parser for general descriptors, which are known to start with 00:00, as they are input at the entry of the inpacient at ICU
def parse_descriptor (lines, descriptor):
    for line in lines:
        if re.match(r'(0?0:0?0),('+descriptor+')', line):
            return decimal.Decimal(line.split(',')[2].strip()) if descriptor != "RecordID" else line.split(',')[2].strip()
    return ''

# parser for time series descriptors, where the time in which it was collected matters
def parse_time_series(lines, descriptor):
    time_series = []
    for line in lines:
        if re.match(r'(\d?\d:\d?\d),('+descriptor+')', line):
            time_series.append((line.split(',')[0], decimal.Decimal(line.split(',')[2].strip())))
    return time_series

# easy way to mount time series descriptors array
def find_time_series_descriptors(set, set_path):
    time_series_descriptors = []
    for file in set:
        for line in open(os.path.join(set_path, file), 'r').read().split('\n'):
            if line != '' and not re.match(r'(RecordID)|(Age)|(Gender)|(Height)|(ICUType)|(Weight)'
                            , line.split(',')[1]) and line.split(',')[1] not in time_series_descriptors:
                time_series_descriptors.append(line.split(',')[1])
    return time_series_descriptors
