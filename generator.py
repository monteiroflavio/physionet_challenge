import os
import parser
import math
import decimal
from math import log
from register import Register
import time_series_summarizer
import pandas as pd

decimal.getcontext().prec = 3
reference_values = pd.read_csv('reference_values.csv', delimiter=',', header=0)
reference_values.set_index('columns', inplace=True)

general_descriptors = ['RecordID', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']
time_series_descriptors = ['FiO2', 'MechVent', 'DiasABP', 'GCS', 'HR'
                          , 'SysABP', 'Temp', 'MAP', 'pH', 'PaCO2', 'PaO2', 'K'
                          , 'Lactate', 'Urine', 'NISysABP', 'SaO2', 'Albumin', 'ALP'
                          , 'ALT', 'AST', 'Bilirubin', 'BUN', 'Cholesterol', 'Creatinine'
                          , 'Glucose', 'HCO3', 'HCT', 'Mg', 'Platelets', 'Na', 'WBC'
                          , 'NIDiasABP', 'NIMAP', 'RespRate', 'TroponinI', 'TroponinT']

decimal.getcontext().prec = 3
def generate_register (file):
    categorical = {}
    time_series = {}
    
    for descriptor in general_descriptors:
        categorical[descriptor] = parser.parse_descriptor(file, descriptor)

    for descriptor in time_series_descriptors:
        time_series[descriptor] = parser.parse_time_series(file, descriptor)
        if descriptor.split('_')[0] != 'MechVent':
            time_series[descriptor] = remove_outliers(descriptor, time_series[descriptor], 10)
        
    return Register(categorical, time_series)

def remove_outliers(column, values, num_sigmas):
    to_remove = []
    for i in range(len(values)):
        if ((values[i][1] < 0)
            or (reference_values.loc[column, "apply_log"] == 1
                and log(float(values[i][1])+1) > reference_values.loc[column, "ln_mean"] + num_sigmas*reference_values.loc[column, "ln_sd"])
            or (reference_values.loc[column, "apply_log"] == 0
                and float(values[i][1]) > reference_values.loc[column, "mean"] + num_sigmas*reference_values.loc[column, "sd"])):
            to_remove.append(values[i])
    values = [value for value in values if value not in to_remove]
    return values

def generateCSVMatrix (set_name, completionMethod, interval):
    file = open(set_name+'.csv', 'w')

    writeHeader(file, interval)
    
    set_path = os.path.join(os.path.join(os.getcwd(), 'sets'), set_name)
    set = [file for file in os.listdir(set_path) if os.path.isfile(os.path.join(set_path, file))]

    for filename in set:
        a_file = open(os.path.join(set_path, filename), 'r').read().split('\n')
        write_register(generate_register(a_file), file, interval, completionMethod)
        
    file.close()
        
def writeHeader (file, interval):
    header = ''
    sorted_general = sorted(general_descriptors)
    for i in range(0, len(sorted_general)):
        header += sorted_general[i]+','
        
    sorted_time_series = sorted([time_serie for time_serie in time_series_descriptors if time_serie != 'MechVent'])
    for i in range(0, len(sorted_time_series)):
        for j in range(0, 48 // interval):
            header += sorted_time_series[i]+'_'+str(j+1)+','
        header += sorted_time_series[i]+'_mean,'
        #header += sorted_time_series[i]+'_var,'
    header += 'MechVent,'
    header += 'SAPS-I,'
    header += 'SOFA,'
    header += 'outcome\n'
    file.write(header)

def write_register (register, file, interval, completionMethod):
    line = ''
    for key in sorted(register.categorical):
        value = register.categorical[key]
        line += '{:.3f}'.format(float(value))+',' if value != -1 else 'NA,'
        
    for key in sorted([time_serie for time_serie in register.time_series if time_serie != 'MechVent']):
        results = time_series_summarizer.intervalTransformer(register.time_series[key],
                                        interval, completionMethod)
        for result in results:
            line += '{:.3f}'.format(float(result))+',' if result > 0 else 'NA,'
        mean = time_series_summarizer.calculateMean(register.time_series[key])
        #var = time_series_summarizer.calculateVariance(register.time_series[key])
        line += '{:.3f}'.format(float(mean))+',' if float(mean) > 0 else 'NA,'
        #line += str(var)+',' if var > 0 else 'NA,'
    line += '1,' if len(register.time_series['MechVent']) > 0 else '0,'
    line += find_extra_info(str(register.categorical['RecordID']),'saps')+',' if float(find_extra_info(str(register.categorical['RecordID']),'saps')) >= 0 else 'NA,'
    line += find_extra_info(str(register.categorical['RecordID']),'sofa')+',' if float(find_extra_info(str(register.categorical['RecordID']),'sofa')) >= 0 else 'NA,'
    line += find_extra_info(str(register.categorical['RecordID']),'outcome')+'\n'
    file.write(line)

def find_extra_info (recordID, info_type):
    file = open(os.path.join(os.path.join(os.getcwd(), 'sets'), 'OutcomesA.txt'), 'r')
    lines = file.read().split('\n')
    file.close()
    for line in lines:
        if line.split(',')[0] != '' and line.split(',')[0] == recordID:
            return {
                'saps':line.split(',')[1]
                , 'sofa':line.split(',')[2]
                , 'outcome':line.split(',')[5]
            }[info_type]
