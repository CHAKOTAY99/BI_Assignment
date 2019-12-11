import numpy as np

# Training Set 2^5 = 32 ABCDE = ¬ACD #
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

# Weights
wIn = np.zeros(5, 4)
wOut = np.zeros(4, 3)

# Target and Input
input = np.array([0, 0, 0, 0, 0])
target = np.array([1, 0, 0])

# These are for the input - outH is the sigmoid of netH
netH = np.zero(1, 4)
outH = np.zero(1, 4)

# These are for the output - outO is the sigmoid of netO
netO = np.zero(1, 3)
outO = np.zero(1, 3)

def initilizeWeights(weight):

