import random
import datetime
import genetic

def test_Hello_World():
    target = "Hallo Welt!"
    guess_password(target)

def guess_password(target):
    geneSet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."
    startTime = datetime.datetime.now()

    def fnGetFitness(genes):
        return get_fitness(genes, target)

    def fnDisplay(genes):
        display(genes, target, startTime)

    optimalFitness = len(target)
    genetic.get_best(fnGetFitness, len(target), optimalFitness, geneset, fnDisplay)

def display(genes, target, startTime):
    timeDiff = datetime.datetime.now() - startTime
    fitness = get_fitness(guess, target)
    print("{0}\t{1}\t{2}".format(guess, fitness, str(timeDiff)))

def get_fitness(guess):
    return sum(1 for expected, actual in zip(target, guess)
                if expected == actual)




random.seed(42)
startTime = datetime.datetime.now()
