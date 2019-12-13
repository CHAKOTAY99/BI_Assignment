import numpy as np

# Training Set 2^5 = 32 ABCDE = ¬ACD # : 80% of 32 = 26, 20% = 6
trainingSet = np.array([["00000", "100"],
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

# Weights random assignment between -1 and 1
wIn = np.random.uniform(low=-1, high=1, size=(5, 4))
wOut = np.random.uniform(low=-1, high=1, size=(4, 3))

# Target and Input
# input = np.array([0, 0, 0, 0, 0])
# target = np.array([1, 0, 0])

# Input - outH is the sigmoid of netH
# netH = np.zeros((1, 4)).astype(np.float64)
# outH = np.zeros((1, 4)).astype(np.float64)

# Output - outO is the sigmoid of netO
# netO = np.zeros((1, 3)).astype(np.float64)
# outO = np.zeros((1, 3)).astype(np.float64)

# Error storage
# errorStore = np.array([0, 0, 0], dtype=np.float64)

# Facts per epoch storage; epoc number,good fact, bad fact
epochStorage = [0, 0, 0]


def setInputAndTarget(tSet):
    inputString = tSet[0, 0]
    targetString = tSet[0, 1]
    inputMatrix = np.array([0, 0, 0, 0, 0])
    targetMatrix = np.array([0, 0, 0])
    i = 0
    x = 0
    for aNumber in inputString:
        inputMatrix[i] = aNumber
        i = i + 1
    for aNumber in targetString:
        targetMatrix[x] = aNumber
        x = x + 1
    print("input\n", inputMatrix)
    print("target\n", targetMatrix)
    return inputMatrix, targetMatrix


def feedforward(feed, goal):
    netH = np.dot(feed, wIn).astype(np.float64)
    outH = sigmoid(netH)
    netO = np.dot(outH, wOut).astype(np.float64)
    outO = sigmoid(netO)
    errorList = [0, 0, 0]
    i = 0
    for eachNumber in outO:
        newNum = np.subtract(goal[i], eachNumber).astype(np.float64)
        errorList[i] = newNum
        i = i + 1

    print("netH:\n", netH)
    print("outH:\n", outH)
    print("netO:\n", netO)
    print("outO:\n", outO)
    print("Error Numbers:\n", errorList)

    return outO, outH, errorList


def checkError(errorList):
    mu = 0.2
    for error in errorList:
        if error > mu:
            return False
    return True


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mainFunction():
    i = 0
    while True:
        # A fact is 1 training set of input and target
        for eachSample in trainingSet:
            itResult = setInputAndTarget(eachSample)
            result = feedforward(itResult[0], itResult[1])
            passFail = checkError(result[1])
            if passFail == False:
                # fact failed and call EBP
                epochStorage[i, 2] += 1
                backPropogation(result[0], result[1], itResult[1])
            elif passFail == True:
                # fact passed and do nothing
                epochStorage[i, 1] += 1
            # End of Fact
        #End of Epoch
        i = i + 1
        if i == 1000:
            ## If we reached the 1000 epoch limit just stop it
            break;
        elif epochStorage[i, 2] == 0:
            ## No errors - great you can stop
            break


def deltaOFormula(outO, targetList):
    i = 0
    deltaO = np.array([0, 0, 0])
    for entry in outO:
        deltaO[i] = entry*(1-entry)*(targetList[i]-entry)
        print("delta\n", deltaO[i])
        i = i + 1
    return deltaO


def sigmaDelta(deltaO, index):
    sum = 0
    x = 0
    for delta in deltaO:
        sum += delta * wOut[index, x]
        x = x + 1

    return sum

def deltaHFormula(deltaO, outH):
    i = 0
    deltaH = np.array([0, 0, 0])
    for entry in outH:
        deltaH[i] = entry*(1-entry)*sigmaDelta(deltaO, i)
        print("delta\n", deltaO[i])
        i = i + 1
    return deltaH


def wOCalc(deltaList, outO):
    eta = 0.2
    i = 0
    wAdj = np.array([0, 0, 0], dtype=np.float64)
    for delta in deltaList:
        wAdj[i] = eta * delta * outO[i]
        i = i + 1
    return wAdj


def wInCalc(deltaList, outH):
    eta = 0.2
    i = 0
    wAdj = np.array([0, 0, 0, 0], dtype=np.float64)
    for delta in deltaList:
        wAdj[i] = eta* delta * outH[i]
        i = i + 1
    return wAdj


def wOAdjust(adj):
    for i in range(4):
        for x in range(3):
            wOut[i, x] += adj[x]

def wInAdjust(adj):
    for i in range(5):
        for x in range(4):
            wIn[i, x] += adj[x]


def backPropogation(outO, outH, targetList):
    eta = 0.2
    # Change weights of output - WOut
    deltaO = deltaOFormula(outO, targetList)
    wOChange = wOCalc(deltaO, outO)
    wOAdjust(wOChange)
    # Change weights of hidden - wIn
    deltaH = deltaHFormula(deltaO, outH)
    wInChange = wInCalc(deltaH, outH)
    wInAdjust(wInChange)



# print("wOut START\n", wOut)
# itResult = setInputAndTarget(trainingSet)
# input = itResult[0]
# target = itResult[1]
# result = feedforward(input, target)
# backPropogation(result[0], result[1], target)
# print("wOut END", wOut)
# print("wIn END", wIn)
# print("Facts:\n", epochStorage)
mainFunction()
