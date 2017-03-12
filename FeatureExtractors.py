import util
import numpy as np
from DataSet import CLASSSET


SYMMETRYTHRESHOLD = 0.2

class FeatureExtractor:
    def __init__(self):
        self.feats = util.Counter()

    def getFeatures(self, datum):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

    # TODO: delete this maybe
    # def getFeatureNames(self):
    #     return ['bias', 'topBottomSymmetry', 'leftRightSymmetry']


class SymmetryExtractor(FeatureExtractor):
    def getFeatures(self, datum):
        """
          Dictionary includes two features that marks the symmetry of
          the image

          Get the four sections of the image and
          sum up the intensity on two sides to compare for symmetry

        """
        self.feats = util.Counter()
        self.feats['bias'] = 1.0

        leftTop = datum.grid[0:3, 0:3]
        rightTop = datum.grid[0:3, 4:7]
        leftBottom = datum.grid[4:7, 0:3]
        rightBotoom = datum.grid[4:7, 4:7]
        upperColorIntensity = np.sum(leftTop+rightTop)
        lowerColorIntensity = np.sum(leftBottom+rightBotoom)
        leftColorIntensity = np.sum(leftTop+leftBottom)
        rightColorIntensity = np.sum(rightTop+rightBotoom)
        # print "leftColorIntensity", upperColorIntensity
        # print "rightColorIntensity", lowerColorIntensity

        lowerUpperRatio = lowerColorIntensity/float(upperColorIntensity)
        leftRightRatio = leftColorIntensity/float(rightColorIntensity)
        # print "ratio", ratio

        self.feats['lowerUpperRatio'] = lowerUpperRatio
        self.feats['leftRightSymmetry'] = leftRightRatio
        return self.feats

class AllGridExtractor(FeatureExtractor):
    def getFeatures(self, datum):
        """
          Dictionary includes 64 features that marks the intensity on
          each grid space of the datum as a single feature
        """

        for i in range(8):
            for j in range(8):
                self.feats[str((i,j))] = datum.grid[i][j]
        self.feats.divideAll(16.)
        self.feats['bias'] = 1.0
        return self.feats

class MaxEntFeatureExtractor(FeatureExtractor):
    V_FOR_SLACK = 16*64.

    def getFeatures(self, datum):
        """
          Dictionary includes 64 features that marks the intensity on
          each grid space of the datum as a single feature
        """

        for c in CLASSSET:
            featureSum = 0
            for i in range(8):
                for j in range(8):
                    self.feats[(str((i, j)),c)] = datum.grid[i][j]
                    featureSum += datum.grid[i][j]

            self.feats['slack',c] = MaxEntFeatureExtractor.V_FOR_SLACK -featureSum
        self.feats.divideAll(self.V_FOR_SLACK)
        return self.feats

class DecisionTreeFeatureExtractor(FeatureExtractor):

    def getFeatures(self, datum):
        """

        :param datum:
        :return: return the features of the brightness of each grid intentensity for the given datum.
        """

        for i in range(8):
            for j in range(8):
                intensity = datum.grid[i][j]
                if intensity < 6:
                    brightness = 0
                elif intensity <11:
                    brightness = 1
                else:
                    brightness = 2
                self.feats[str((i, j))] = brightness

        return self.feats

    def getFeatureValues(self):
        return [0,1,2]

