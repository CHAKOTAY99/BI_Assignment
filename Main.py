import numpy as np
import random
import sys
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)

# Training Set 2^5 = 32 ABCDE = ¬ACD # : 80% of 32 = 26, 20% = 6
dataSet = np.array([["00000", "100"],
                    ["00001", "100"],
                    ["00010", "101"],
                    ["00011", "101"],
                    ["00100", "110"],
                    ["00101", "110"],
                    ["00110", "111"],
                    ["00111", "111"],
                    ["01000", "100"],
                    ["01001", "100"],
                    ["01010", "101"],
                    ["01011", "101"],
                    ["01100", "110"],
                    ["01101", "110"],
                    ["01110", "111"],
                    ["01111", "111"],
                    ["10000", "000"],
                    ["10001", "000"],
                    ["10010", "001"],
                    ["10011", "001"],
                    ["10100", "010"],
                    ["10101", "010"],
                    ["10110", "011"],
                    ["10111", "011"],
                    ["11000", "000"],
                    ["11001", "000"],
                    ["11010", "001"],
                    ["11011", "001"],
                    ["11100", "010"],
                    ["11101", "010"],
                    ["11110", "011"],
                    ["11111", "011"],
                    ])


def randomTrainingSet(dataSet):
    array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    random.shuffle(dataSet)
    trainingSet = dataSet[:26]
    testSet = dataSet[26:]
    return trainingSet, testSet


def setInputAndTarget(fact):
    inputString = fact[0]
    targetString = fact[1]
    inputMatrix = np.array([0, 0, 0, 0, 0], dtype=np.float64)
    targetMatrix = np.array([0, 0, 0], dtype=np.float64)
    i = 0
    x = 0
    for aNumber in inputString:
        inputMatrix[i] = aNumber
        i = i + 1
    for aNumber in targetString:
        targetMatrix[x] = aNumber
        x = x + 1
    return inputMatrix, targetMatrix


def feedforward(input, target, wIn, wOut):
    netH = np.dot(input, wIn).astype(np.float64)
    outH = sigmoid(netH)
    netO = np.dot(outH, wOut).astype(np.float64)
    outO = sigmoid(netO)
    errorList = np.array([0, 0, 0], dtype=np.float64)
    i = 0
    for eachNumber in outO:
        errorList[i] = np.subtract(target[i], eachNumber)
        i = i + 1

    return outO, outH, errorList


def checkError(errorList):
    mu = 0.2
    for error in errorList:
        if abs(error) > mu:
            return False
    return True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mainFunction():
    # Weights random assignment between -1 and 1
    wIn = np.random.uniform(low=-1, high=1, size=(5, 4))
    wOut = np.random.uniform(low=-1, high=1, size=(4, 3))
    resultSet = randomTrainingSet(dataSet)
    totEpoch = np.zeros((1000,3), dtype=np.float64)
    i = 0
    while True:
        ## epochStorage good - bad fact %tage
        totEpoch[i, 0] = i
        # EPOCH START
        for fact in resultSet[0]:
            ## setInputAndTarget OUTPUT: inputMarix and targetMatrix in that order INPUT: fact
            factResult = setInputAndTarget(fact)
            ## feedforward OUTPUT: outO, outH and errorList in that order INPUT: input matrix, target matrix, wIn, wOut
            ffResult = feedforward(factResult[0], factResult[1], wIn, wOut)
            passFail = checkError(ffResult[2])
            if passFail == False:
                # fact failed and call EBP
                totEpoch[i, 2] += 1
                ## backPropogation INPUT: outO, outH, targetList, wIn, wOut
                backPropogation(ffResult[0], ffResult[1], factResult[1], factResult[0], wIn, wOut)
            elif passFail == True:
                # fact passed and do nothing
                totEpoch[i, 1] += 1
            # End of Fact
        totEpoch[i, 1] = (totEpoch[i, 1] / 26) * 100
        totEpoch[i, 2] = (totEpoch[i, 2] / 26) * 100
        # End of Epoch
        if i == 1000:
            ## If we reached the 1000 epoch limit just stop it
            totEpoch = np.delete(totEpoch, np.s_[i+1:], axis=0)
            return totEpoch
        elif totEpoch[i, 2] == 0:
            ## No errors - great you can stop
            totEpoch = np.delete(totEpoch, np.s_[i+1:], axis=0)
            return totEpoch
        # Add new line to epoch storage
        i = i + 1


def deltaOFormula(outO, targetList):
    i = 0
    deltaO = np.array([0, 0, 0], dtype=np.float64)
    for entry in outO:
        deltaO[i] = entry * (1 - entry) * (targetList[i] - entry)
        i = i + 1
    return deltaO


def sigmaDelta(deltaO, index, wOut):
    sum = 0
    x = 0
    for delta in deltaO:
        sum += (delta * wOut[index, x])
        x = x + 1
    return sum


def deltaHFormula(deltaO, outH, wOut):
    i = 0
    deltaH = np.array([0, 0, 0, 0], dtype=np.float64)
    for entry in outH:
        deltaH[i] = (entry * (1 - entry) * (sigmaDelta(deltaO, i, wOut)))
        i = i + 1
    return deltaH


def wOCalc(deltaList, outH, wOut):
    eta = 0.2
    i = 0
    for output in outH:
        x = 0
        for delta in deltaList:
            wOut[i][x] += eta * delta * output
            x = x + 1
        i = i + 1


def wInCalc(deltaList, inputList, wIn):
    eta = 0.2
    i = 0
    for input in inputList:
        x = 0
        for delta in deltaList:
            wIn[i][x] += eta * delta * input
            x = x + 1
        i = i + 1


def backPropogation(outO, outH, targetList, inputList, wIn, wOut):
    # Change weights of output - WOut
    deltaO = deltaOFormula(outO, targetList)
    wOCalc(deltaO, outH, wOut)
    # Change weights of hidden - wIn
    deltaH = deltaHFormula(deltaO, outH, wOut)
    wInCalc(deltaH, inputList, wIn)

def plotGraph(data):
    reference = pd.DataFrame(data=data[0:, 1:],
                 index=data[0:, 0],
                 columns=data[0, 1:])
    reference.columns = ['GoodFacts', 'BadFacts']
    print(reference)

epochStorage = mainFunction()
print(epochStorage)
print("-----------------------")
plotGraph(epochStorage)