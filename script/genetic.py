import random
from script.phrase1 import Phrase,target,popSize,lot_size
from script.helpers import summarize
from sklearn.preprocessing import MinMaxScaler
population = []
bestScore = -111222
generation = 1
siz = 4096
for i in range(popSize):
    population.append(Phrase())

child_set = set([])
preper = 0
while generation<50:
    for i in range(popSize):
        population[i].getFitness()
        pre = len(child_set)
        child_set.add(population[i].getContents())
        now = len(child_set)
        has = ''
        if now != pre:
            per = (now / siz)*50
            # per = int(per)
            per = int(per)
            if preper != per:
                for j in range(int(per)):
                    has += '='
                has+='>'
                for j in range(int(50 - per)):
                    has += '.'
                print(has+str(len(child_set))+"/"+str(siz))
            preper = int(per)

        if int(population[i].score) > bestScore:
            bestScore = population[i].score
            summarize(generation, population[i].getContents(), bestScore)
    # create the mating pool for the next generation
    matingPool = []

    # clear the population array, but save the parents
    parents = population[:]
    population = []

    matp= []
    # for each one of the parents, add it to the mating pool more often if
    # its fitness is highero


    '''
    print("gen$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ "+str(generation)+" " + str(len(child_set)))
    for i in range(len(parents)):
        print(str(parents_sorted[i].score)+" "+str(parents_sorted[i].getContents()))
    print("after sort$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    '''
    parents_sorted = parents
    parents_sorted.sort(key=lambda x: x.score, reverse=True)
    for i in range(len(parents_sorted)):
        matp.append([parents_sorted[i].score,0])
    sc = MinMaxScaler(feature_range=(0,5));
    sc.fit(matp)
    # print(sc.data_max_(matp))
    norm = sc.transform(matp)
    # print(len(norm))
    # print(norm)
    # print(parents_sorted)
    for i in range(popSize):
        # for j in range(int(parents[i].score)):
        if len(matingPool) < popSize:
            fre = int(norm[i][0])
            # fre = fre / lot_size
            for k in range(2):
                if len(matingPool) < popSize:
                    matingPool.append(parents_sorted[i])
                else:
                    break
        else: break

    # build the next generation
    for i in range(popSize):

        # arbitrarily choose two parents from the mating pool
        parentA = random.choice(matingPool)
        parentB = random.choice(matingPool)

        # crossover/breed those two parents together
        child = parentA.crossover(parentB)

        # small chance that some characters in the child may mutate
        child.mutate()

        # add the child to the next generation's population
        population.append(child)

    # done assessing the current generation
    print(generation)
    generation += 1
