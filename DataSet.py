from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import random

DIMENSION = 8
CLASSSET = ['0','1','2','3','4','5','6','7','8','9']

class DataSet():
    """
    DataSet that functions essentially like a list. The initializer reads the data file and
    generate a list of Datum objects representing the instances.
    """
    def __init__(self, filename):
        """
        Read a file and generate a DataSet object.
        :param filename: the filename.
        """
        self.data = []
        with open(filename, 'r') as f:
            for line in f:
                if line[0] != '#':
                    # not a comment line
                    numbers = line.rstrip('\n').split(',')
                    label = numbers[-1]
                    grid = np.array(numbers[:-1]).astype(np.int)
                    # reshape the data to a 8x8 grid
                    grid = grid.reshape((DIMENSION, DIMENSION))
                    label = numbers[-1]
                    self.data.append(Datum(grid, label))

    def plotRandom(self):
        """
        Plot a random instance in the dataset and print the label to the screen.
        :return:
        """
        dat = random.choice(self.data)
        plot = dat.draw()
        print dat.label
        plt.show(plot)

    def __iter__(self):
        return iter(self.data)
    
    def __getitem__(self, key):
        return self.data[key]

    def __len__(self):
        return len(self.data)


class Datum(object):
    """
    Datum object holds one data sample.
    """
    def __init__(self, grid, label):
        """

        :param grid: the 8x8 matrix of integers from the preprocessed data.
        :param label: the label of the data poit.
        """
        self.grid = grid
        self.label = label

    def __str__(self):
        return 'grid: ' + str(self.grid) +' label:' + self.label

    def draw(self):
        return plt.imshow(self.grid, norm=colors.Normalize(vmin=0., vmax=16.), cmap='gray', interpolation='nearest')
