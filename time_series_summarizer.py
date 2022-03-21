from math import floor
import decimal
import imputation_methods
import numpy as np

decimal.getcontext().prec = 3

def intervalTransformer (list, intervalLenght, completionMethod):
    if intervalLenght > 48:
        intervalLenght = 48
    if 48 % intervalLenght > 0:
        return "Not a valid interval"
    intervals = [[] for i in range(0, 48 // intervalLenght)]
    for i in range(0, len(list)):
        hour = int(floor(int(list[i][0].split(':')[0])))
        index = hour // intervalLenght if hour < 48 else hour // intervalLenght - 1
        intervals[index].append(list[i][1])
    for i in range(0, len(intervals)):
        intervals[i] = decimal.Decimal(sum(intervals[i]) / len(intervals[i])) if len(intervals[i]) > 0 else 0
    return {
        1: imputation_methods.neighbourCompletion(intervals),
        2: imputation_methods.meanCompletion(intervals),
        3: imputation_methods.hibridCompletion(intervals)
    }[completionMethod]

def calculateMean (list):
    #return decimal.Decimal(sum([float(value[1]) for value in list]) / len(list)) if len(list) > 0 else 0
    return decimal.Decimal(np.mean(np.array([value[1] for value in list])))

def calculateVariance (list):
    return decimal.Decimal(np.var(np.array([value[1] for value in list])))
