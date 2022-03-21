def binarizeConfusionMatrix(confusionMatrix):
    cm = [[0 for x in range(2)] for x in range(2)]
    for i in range(len(confusionMatrix)):
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        for j in range(len(confusionMatrix)):
            if j == i:
                tp = confusionMatrix[i][j]
            else:
                fn += confusionMatrix[i][j]
                fp += confusionMatrix[j][i]
        for k in range (len(confusionMatrix)):
            for m in range (len(confusionMatrix)):
                if (k and m) != i:
                    tn += confusionMatrix[k][m]
        cm[0][0] += tp
        cm[0][1] += fn
        cm[1][0] += fp
        cm[1][1] += tn
    return cm

def binarizeLabels(array, positiveLabels):
    return [0 if value in positiveLabels else 1 for value in array]
        
def binaryConfusionMatrix(confusionMatrix, positiveRange):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for i in range(positiveRange+1):
        for j in range(positiveRange+1):
            tp += confusionMatrix[i][j]
        for j in range(positiveRange+1, len(confusionMatrix)):
            fn += confusionMatrix[i][j]
    for i in range(positiveRange+1, len(confusionMatrix)):
        for j in range(positiveRange+1):
            fp += confusionMatrix[i][j]
        for j in range(positiveRange+1, len(confusionMatrix)):
            tn += confusionMatrix[i][j]
    return [[tp,fn],[fp,tn]]

def oneVsAllConfusionMatrix(confusionMatrix, pos):
    tp, fp, fn, tn = [0 for i in range(4)]
    for i in range(len(confusionMatrix)):
        for j in range(len(confusionMatrix)):
            if i == pos:
                if j == pos:
                    tp = confusionMatrix[i][j]
                else:
                    fp += confusionMatrix[i][j]
            elif j == pos and i != pos:
                fn += confusionMatrix[i][j]
            else:
                tn += confusionMatrix[i][j]
    return [[tp, fp],[fn, tn]]
