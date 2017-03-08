from matplotlib import pyplot as plt
from matplotlib import colors
import random

DIMENSION = 8
class DataSet():
    def __init__(self, filename):
        self.data = []
        with open(filename, 'r') as f:
            for line in f:
                if line[0] != '#':
                    # not a comment line
                    numbers = line.rstrip('\n').split(',')
                    numbersIterator = iter(numbers)
                    grid = []
                    # reshape the data to a 8x8 grid
                    for i in xrange(DIMENSION):
                        row = []
                        for j in xrange(DIMENSION):
                            row.append(int(next(numbersIterator)))
                        grid.append(row)
                    # print grid, next(numbersIterator), numbers[-1]
                    label = numbers[-1]
                    self.data.append(Datum(grid, label))

    def plotRandom(self):
        dat = random.choice(self.data)
        plot = dat.draw()
        print dat.label
        plt.show(plot)

    def __iter__(self):
        return iter(self.data)
    
    def __getitem__(self, key):
        return iter(self.data[key])


class Datum(object):
    def __init__(self, grid, label):
        """ Datum object holds one data sample.
        """
        self.grid = grid
        self.label = label

    def __str__(self):
        return 'grid: ' + str(self.grid) +' label:' + self.label

    def draw(self):
        return plt.imshow(self.grid, norm=colors.Normalize(vmin=0., vmax=16.), cmap='gray', interpolation='nearest')
