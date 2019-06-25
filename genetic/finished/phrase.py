####################################
# phrase.py                        #
#                                  #
# Defines Phrase class for genetic #
#                                  #
# Doug Lloyd                       #
# March 15, 2019                   #
# CS50                             #
####################################

import random
from cs50 import get_string
import pandas as pd
import statistics
df1 = pd.read_csv("/root/alpha/git/mine/data_science/data/01JAN/SBIN.csv",header=None)
df = df1[6]
avg = statistics.mean(df)
# ask the user for a target string
# target = get_string("What target do you want to match? ")
target = 2

class Phrase:

    # constructor method
    def __init__(self):
        self.characters = []
        # append len(target) number of randomly chosen printable ASCII chars
        for i in range(target):
            # if i % 3 != 0:
            character = chr(random.choice(range(48,50)))
            self.characters.append(character)
        self.r1_avg_p = statistics.mean(df)
            # else:
            #     self.characters.append('0')

    # render the character array as a string instead of an array of chars
    def getContents(self):
        return ''.join(self.characters)

    # score the current entity's fitness by counting matches to target
    def getFitness(self):
        self.score = 0
        self.shares = 0
        self.buy = 0
        self.buy_price = 0
        self.sell = 0
        self.sell_price = 0
        # self.r1_avg
        # sel
        # for i in range(len(self.characters)):
        for j in range(len(df)):
            if df[j] > self.r1_avg_p:
                # self.characters[0] = '1'
                if self.characters[1] == '0':
                    if self.buy == 1 and self.sell == 0:
                        self.buy = 0
                        self.score += (df[j] - self.buy_price)
                    elif self.buy == 0 and self.buy == 0:
                        self.sell = 1
                        self.sell_price = df[j]
                else:#if self.characters[2] == '1':
                    if self.sell == 1  and self.buy == 0:
                        self.sell = 0
                        self.score -= (df[j] - self.sell_price)
                    elif self.sell == 0 and self.buy == 0:
                        self.buy = 1
                        self.buy_price = df[j]
            else:
                # self.characters[0] = '0'
                if self.characters[0] == '0':
                    if self.buy == 1  and self.sell == 0:
                        self.buy = 0
                        self.score += (df[j] - self.buy_price)
                    elif self.buy == 0  and self.sell == 0:
                        self.sell = 1
                        self.sell_price = df[j]
                else:#if self.characters[2] == '1':
                    if self.sell == 1  and self.buy == 0:
                        self.sell = 0
                        self.score -= (df[j] - self.sell_price)
                    elif self.sell == 0  and self.buy == 0:
                        self.buy = 1
                        self.buy_price = df[j]
        if self.sell == 1 and self.buy == 0:
            self.sell = 0
            self.score -= (df[j] - self.sell_price)
        elif self.buy == 1 and self.sell == 0:
            self.buy = 0
            self.score += (df[j] - self.buy_price)

            # print("sell:"+str(self.sell)+" buy:"+str(self.buy)+" sell_p: "+str(self.sell_price)+" buy_p: "+str(self.buy_price))
    # create a child of two members of the current generation
    def crossover(self, partner):

        # create a spot for the characters to go
        child = Phrase()

        # flip a coin for each character, selecting from one parent each time
        for i in range(len(self.characters)):
            if i % 3 != 0:
                if random.random() < 0.5:
                    child.characters[i] = self.characters[i]
                else:
                    child.characters[i] = partner.characters[i]

        return child

    # some portion of the time, need some characters to randomly change
    def mutate(self):

        # less than 1% of the time, change a character into something else
        for i in range(len(self.characters)):
            # if i % 3 != 0:
            if random.random() < 0.01:
                self.characters[i] = chr(random.choice(range(48,50)))
