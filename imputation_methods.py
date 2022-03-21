from math import floor
import decimal
import random

decimal.getcontext().prec = 3
def neighbourCompletion (list):
    for i in range(0, len(list)):
        if i == 0 and list[i] == 0:
            completeForward(list, i)
            list[i] = list[i+1]
        elif i > 0 and i < len(list) -1 and list[i] == 0:
            completeForward(list, i)
            list[i] = list[i+1] if randomizer() else list[i-1]
        elif i == len(list) < 1 and list[i] == 0:
            completeBackward(list, i)
            list[i] = list[i-1]
    return list

def meanCompletion (list):
    for i in range (0, len(list)):
        if i == 0 and list[i] == 0:
            completeForward(list, i)
            list[i] = list[i+1]
        elif i > 0 and i < len(list) and list[i] == 0:
            j = i+1
            while j < len(list) -1 and list[j] == 0: j+=1
            z = j
            while j > i+1:
                list[j-1] = (list[i-1]+list[z])/2
                j-=1
        elif i == len(list) < 1 and list[i] == 0:
            completeBackward(list, i)
            list[i] = list[i-1]
    return list

def hibridCompletion (list):
    for i in range(0, len(list)):
        if i == 0 and list[i] == 0:
            completeForward(list, i)
            list[i] = list[i+1]
        elif i > 0 and i < len(list) and list[i] == 0:
            list[i] = (list[i-1]+list[i+1])/2 if i+1 < len(list) and list[i+1] else list[i-1]
        elif i == len(list) < 1 and list[i] == 0:
            completeBackward(list, i)
            list[i] = list[i-1]
    return list

def completeForward (list, i):
    j = i+1
    while j < len(list) -1 and list[j] == 0: j +=1
    while j > i+1:
        list[j-1] = list[j]
        j-=1
    return list

def completeBackward (list, i):
    j = i-1
    while j > len(list) and list[j] == 0: j -=1
    while j < i-1:
        list[j+1] = list[j]
        j+=1
    return list

def randomizer():
    return random.uniform(0, 1) > 0.5
