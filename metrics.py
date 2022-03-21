def calculateScore1 (confusionMatrix):
    return min(calculateSensitivity(confusionMatrix[0][0]
    	, confusionMatrix[1][0])
		, calculatePositivePredictibility(confusionMatrix[0][0]
        , confusionMatrix[0][1]))

def calculateSensitivity (truePositive, falseNegative):
    return truePositive/(truePositive+falseNegative)

def calculatePositivePredictibility (truePositive, falsePositive):
    return truePositive/(truePositive+falsePositive)
