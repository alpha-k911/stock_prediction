import random
import numpy,math
import pandas as pd
import statistics
import talib
df1 = pd.read_csv("/root/alpha/git/mine/data_science/data/03JAN/SBIN.txt",sep=",",header=None)
dfc = pd.DataFrame(df1[[6, 7]])
dfc = dfc[1:180]
dfc = pd.DataFrame(dfc)
dfc.reset_index(inplace=True)
df = dfc[6]
dfv = dfc[7]
# avg = statistics.mean(df)
target = 14
popSize = 100
c1 = 0
lot_size = 500

class Phrase:
    def __init__(self):
        self.characters = []
        for i in range(target):
            character = chr(random.choice(range(48,50)))
            self.characters.append(character)
        self.r1_avg_p = statistics.mean(df)
        self.count0 = 0

    def set(self, contents):
        self.characters = []
        # append len(target) number of randomly chosen printable ASCII chars
        for i in range(len(contents)):
            character = contents[i]
            self.characters.append(character)
    def getContents(self):
        return ''.join(self.characters)
    def getFitness(self):
        self.score = 0
        self.shares = 0
        self.buy = 0
        self.buy_price = 0
        self.sell = 0
        self.sell_price = 0
        self.transaction = 0
        # self.sl = 0
        for p in range(len(df) - 21):
            j = p + 20
            self.count0 = 0
            self.r1(j)
            self.r2(j)
            self.r3(j)
            self.r4(j)
            self.r5(j)
            self.r6(j)
            self.r7(j)
            brokerage = 40

            if self.buy == 1 and self.sell == 0 and ((df[j] - self.buy_price) / self.buy_price) * 100 < (-1):

                # print("stoploss triggered")
                self.buy = 0
                if ((df[j] + self.buy_price) * lot_size * 0.06 < 40):
                    brokerage = (df[j] + self.buy_price) * lot_size * 0.06
                self.score += ((df[j] - self.buy_price) * lot_size - brokerage)
                self.transaction += 1

            if self.buy == 0 and self.sell == 1 and ((self.sell_price - df[j]) / self.sell_price) * 100 < (-1):
                # print("stoploss triggered")
                self.sell = 0
                if ((df[j] + self.sell_price) * lot_size * 0.06 < 40):
                    brokerage = (df[j] + self.sell_price) * lot_size * 0.06
                self.score = self.score + ((self.sell_price - df[j]) * lot_size) - brokerage
                self.transaction += 1

            if self.count0 > (int(target / 4)):

                if self.buy == 1 and self.sell == 0:
                    self.buy = 0
                    if ((df[j] + self.buy_price) * lot_size * 0.06 < 40):
                        brokerage = (df[j] + self.buy_price) * lot_size * 0.06
                    self.score += ((df[j] - self.buy_price) * lot_size) - brokerage
                    self.transaction += 1

                elif self.sell == 0 and self.buy == 0:
                    self.sell = 1
                    self.sell_price = df[j]


            elif self.count0 < (int(target / 4)):  # if self.characters[2] == '1':

                if self.sell == 1 and self.buy == 0:
                    self.sell = 0
                    if ((df[j] + self.sell_price) * lot_size * 0.06 < 40):
                        brokerage = (df[j] + self.sell_price) * lot_size * 0.06
                    self.score = self.score + ((self.sell_price - df[j]) * lot_size) - brokerage
                    self.transaction += 1


                elif self.sell == 0 and self.buy == 0:
                    self.buy = 1
                    self.buy_price = df[j]

        brokerage = 40
        if self.sell == 1 and self.buy == 0:
            self.sell = 0
            if ((df[j] + self.sell_price) * lot_size * 0.06 < 40):
                brokerage = (df[j] + self.sell_price) * lot_size * 0.06
            self.score = self.score + ((self.sell_price - df[j]) * lot_size) - brokerage
            self.transaction += 1;


        elif self.buy == 1 and self.sell == 0:
            self.buy = 0
            if ((df[j] + self.buy_price) * lot_size * 0.06 < 40):
                brokerage = (df[j] + self.buy_price) * lot_size * 0.06
            self.score += ((df[j] - self.buy_price) * lot_size - brokerage)
            self.transaction += 1

            # print("sell:"+str(self.sell)+" buy:"+str(self.buy)+" sell_p: "+str(self.sell_price)+" buy_p: "+str(self.buy_price))
        # create a child of two members of the current generation

    def crossover(self, partner):

        # create a spot for the characters to go
        child = Phrase()
        # flip a coin for each character, selecting from one parent each time
        for i in range(len(self.characters)):
            if random.random() < 0.5:
                child.characters[i] = self.characters[i]
        return child

    # some portion of the time, need some characters to randomly change
    def mutate(self):

        # less than 1% of the time, change a character into something else
        for i in range(len(self.characters)):
            # if i % 3 != 0:
            if random.random() < 0.03:
                self.characters[i] = chr(random.choice(range(48, 50)))

    def r1(self, j):
        if j > 0:
            close_prices = df[0:j]
            volumes = dfv[0:j]
            rocp = talib.ROCP(close_prices, timeperiod=1)
            norm_volumes = (volumes - numpy.mean(volumes)) / math.sqrt(numpy.var(volumes))
            vrocp = talib.ROCP(norm_volumes + numpy.max(norm_volumes) - numpy.min(norm_volumes), timeperiod=1)
            # vrocp = talib.ROCP(volumes, timeperiod=1)
            if rocp[j - 1] != numpy.NaN and vrocp[j - 1] != numpy.NaN:
                pv = rocp[j - 1] * vrocp[j - 1] * 100
                if pv > 0:
                    if rocp[j - 1] > 0 and vrocp[j - 1] > 0:
                        if self.characters[1] == '0':
                            self.count0 += 1
                    elif rocp[j - 1] < 0 and vrocp[j - 1] < 0:
                        if self.characters[0] == '0':
                            self.count0 += 1

    def r2(self, j):
        if j > 0:
            avg = statistics.mean(df[j - 20:j])
            if df[j] > avg:
                if self.characters[3] == '0':
                    self.count0 += 1
            elif df[j] < avg:
                if self.characters[2] == '0':
                    self.count0 += 1

    def r3(self, j):
        if j > 0:
            rocp = talib.ROCP(df[0:j], timeperiod=1)
            if rocp[j - 1] > 0:
                if self.characters[5] == '0':
                    self.count0 += 1
            elif rocp[j - 1] < 0:
                if self.characters[4] == '0':
                    self.count0 += 1

    def r4(self, j):
        if j > 34:
            macd, signal, hist = talib.MACD(df[0:j], fastperiod=12, slowperiod=26, signalperiod=9)
            a = macd[j - 2] - hist[j - 2]
            b = macd[j - 1] - hist[j - 1]
            if a > 0 and b < 0:
                if self.characters[7] == '0':
                    self.count0 += 1
            elif a < 0 and b > 0:
                if self.characters[6] == '0':
                    self.count0 += 1

    def r5(self, j):
        if j > 0:
            upperband, middleband, lowerband = talib.BBANDS(df[0:j], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            if upperband[j - 1] < df[j]:
                if self.characters[9] == '0':
                    self.count0 += 1
            elif lowerband[j - 1] > df[j]:
                if self.characters[8] == '0':
                    self.count0 += 1

    def r6(self, j):
        if j > 0:
            volumes = dfv[0:j]
            norm_volumes = (volumes - numpy.mean(volumes)) / math.sqrt(numpy.var(volumes))
            vrocp = talib.ROCP(norm_volumes + numpy.max(norm_volumes) - numpy.min(norm_volumes), timeperiod=1)
            if vrocp[j - 1] > 0:
                if self.characters[11] == '0':
                    self.count0 += 1
            elif vrocp[j - 1] < 0:
                if self.characters[10] == '0':
                    self.count0 += 1

    def r7(self, j):
        if (j > 20):
            rsi = talib.RSI(df[0:j], timeperiod=12)
            # macd, signal, hist = talib.MACD((df[0:j]), fastperiod=12, slowperiod=26, signalperiod=9)
            # print(str(a) +" "+ str(b))
            if rsi[j - 1] > 70:
                # print(str(a) + str(b))
                if self.characters[13] == '0':
                    self.count0 += 1
            elif rsi[j - 1] < 30:
                # print(str(a) + str(b))
                if self.characters[12] == '0':
                    self.count0 += 1