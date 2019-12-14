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
    # print("input\n", inputMatrix)
    # print("target\n", targetMatrix)
    return inputMatrix, targetMatrix


def feedforward(input, target):
    # print("input and target ", input, target)
    netH = np.dot(input, wIn).astype(np.float64)
    outH = sigmoid(netH)
    netO = np.dot(outH, wOut).astype(np.float64)
    outO = sigmoid(netO)
    errorList = np.array([0, 0, 0], dtype=np.float64)
    i = 0
    for eachNumber in outO:
        newNum = np.subtract(target[i], eachNumber)
        errorList[i] = newNum
        i = i + 1

    # print("netH:\n", netH)
    # print("outH:\n", outH)
    # print("netO:\n", netO)
    # print("outO:\n", outO)
    # print("Error Numbers:\n", errorList)

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
    i = 0
    while True:
        ## tempArray good - bad fact
        tempArray = np.array([i, 0, 0])
        # EPOCH START
        for fact in trainingSet:
            ## setInputAndTarget OUTPUT: inputMarix and targetMatrix in that order INPUT: fact
            factResult = setInputAndTarget(fact)
            ## feedforward OUTPUT: outO, outH and errorList in that order INPUT: input matrix, target matrix
            ffResult = feedforward(factResult[0], factResult[1])
            passFail = checkError(ffResult[2])
            if passFail == False:
                # fact failed and call EBP
                # print("fact fail")
                tempArray[2] += 1
                # print("fail ", fact)
                ## backPropogation INPUT: outO, outH, targetList
                backPropogation(ffResult[0], ffResult[1], factResult[1], factResult[0])
            elif passFail == True:
                # fact passed and do nothing
                # print("fact pass")
                # print("pass ", fact)
                tempArray[1] += 1
            # End of Fact
        #End of Epoch
        print("Temp Array ", tempArray)
        if i == 100000:
            ## If we reached the 1000 epoch limit just stop it
            break
        elif tempArray[2] == 0:
            ## No errors - great you can stop
            break
        # Add new line to epoch storage
        i = i + 1


def deltaOFormula(outO, targetList):
    i = 0
    deltaO = np.array([0, 0, 0], dtype=np.float64)
    for entry in outO:
        deltaO[i] = entry*(1-entry)*(targetList[i]-entry)
        i = i + 1
    return deltaO


def sigmaDelta(deltaO, index):
    sum = 0
    x = 0
    for delta in deltaO:
        value = wOut[index, x]
        sum += (delta * wOut[index, x])
        x = x + 1
    return sum

def deltaHFormula(deltaO, outH):
    i = 0
    deltaH = np.array([0, 0, 0, 0], dtype=np.float64)
    for entry in outH:
        deltaH[i] = (entry*(1-entry)*(sigmaDelta(deltaO, i)))
        i = i + 1
    return deltaH


def wOCalc(deltaList, outO):
    eta = 0.2
    i = 0
    wAdj = np.zeros(3, np.float64)
    for delta in deltaList:
        wAdj[i] = eta * delta * outO[i]
        i = i + 1
    # print("NEW WEIGHT w0: ", wAdj)
    return wAdj


def wInCalc(deltaList, inputList):
    eta = 0.2
    i = 0
    wAdj = np.zeros(4, np.float64)
    for delta in deltaList:
        wAdj[i] = (eta * delta * inputList[i])
        i = i + 1
    # print("NEW WEIGHT wIn: ", wAdj)
    return wAdj


def wOAdjust(adj):
    for i in range(4):
        for x in range(3):
            global wOut
            wOut[i, x] += adj[x]

def wInAdjust(adj):
    for i in range(5):
        for x in range(4):
            global wIn
            wIn[i, x] += adj[x]


def backPropogation(outO, outH, targetList, inputList):
    eta = 0.2
    # Change weights of output - WOut
    deltaO = deltaOFormula(outO, targetList)
    # print("deltaO\n", deltaO)
    wOChange = wOCalc(deltaO, outO)
    wOAdjust(wOChange)
    # Change weights of hidden - wIn
    deltaH = deltaHFormula(deltaO, outH)
    # print("deltaH\n", deltaH)
    wInChange = wInCalc(deltaH, inputList)
    wInAdjust(wInChange)



# print("wIn START\n", wIn)
# print("wOut START\n", wOut)
# itResult = setInputAndTarget(trainingSet)
# result = feedforward(itResult[0], itResult[1])
# backPropogation(result[0], result[1], itResult[1])
# print("Facts:\n", epochStorage)


mainFunction()

# print("wIn END\n", wIn)
# print("wOut END\n", wOut)
# print("EPOCHSTORAGE\n", epochStorage)
