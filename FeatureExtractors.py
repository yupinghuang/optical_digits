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
        feats = util.Counter()

        for i in range(8):
            for j in range(8):
                feats[str((i,j))] = datum.grid[i][j]
        feats.divideAll(16.)
        feats['bias'] = 1.0
        return feats

class MaxEntFeatureExtractor(FeatureExtractor):
    V_FOR_SLACK = 16*64.

    def getFeatures(self, datum):
        """
          Dictionary includes 64 features that marks the intensity on
          each grid space of the datum as a single feature
        """

        feats = util.Counter()
        for c in range(10):
            featureSum = 0
            for i in range(8):
                for j in range(8):
                    feats[(str((i, j)),CLASSSET[c])] = datum.grid[i][j]
                    featureSum += datum.grid[i][j]
            feats['slack',CLASSSET[c]] = MaxEntFeatureExtractor.V_FOR_SLACK -featureSum

        return self.feats


