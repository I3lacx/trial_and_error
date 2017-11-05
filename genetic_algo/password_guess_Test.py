import random
import datetime
import genetic
import unittest

class GuessPasswordTests(unittest.TestCase):
    geneSet = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!.123456789"


    def test_Hello_World(self):
        target = "Hello World!asd asdasdwdawdsgasdfg"
        self.guess_password(target)

    def guess_password(self, target):
        startTime = datetime.datetime.now()

        def fnGetFitness(genes):
            return get_fitness(genes, target)

        def fnDisplay(genes):
            display(genes, target, startTime)

        optimalFitness = len(target)
        best = genetic.get_best(fnGetFitness, len(target), optimalFitness, self.geneSet, fnDisplay)
        self.assertEqual(best, target)

def display(genes, target, startTime):
    timeDiff = datetime.datetime.now() - startTime
    fitness = get_fitness(genes, target)
    print("{0}\t{1}\t{2}".format(genes, fitness, str(timeDiff)))

def get_fitness(guess, target):
    return sum(1 for expected, actual in zip(target, guess)
                if expected == actual)

if __name__ == '__main__':
    unittest.main()
