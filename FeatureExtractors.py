import util
import numpy as np
from DataSet import CLASSSET
import Classfiers


SYMMETRYTHRESHOLD = 0.2

class FeatureExtractor:
    """
    Abstract FeatureExtrator class.
    """
    def getFeatures(self, datum):
        """
          Returns a dict from features to feature values.
        """
        util.raiseNotDefined()


class SymmetryExtractor(FeatureExtractor):
    def getFeatures(self, datum):
        """
          Dictionary includes two features that marks the symmetry of
          the image

          Get the four sections of the image and
          sum up the intensity on two sides to compare for symmetry

        """
        feats = util.Counter()
        feats['bias'] = 1.0

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

        feats['lowerUpperRatio'] = lowerUpperRatio
        feats['leftRightSymmetry'] = leftRightRatio
        return feats

class AllGridExtractor(FeatureExtractor):
    """
    Features based on the intensity of the 8x8 grid for MIRA.
    """
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
    """
    Features based on the intensity of the 8x8 grid for MaxEnt.
    """
    V_FOR_SLACK = 16*64.

    def getFeatures(self, datum):
        """
          Dictionary includes 64 features that marks the intensity on
          each grid space of the datum as a single feature
        """
        feats = util.Counter()
        for c in CLASSSET:
            featureSum = 0
            for i in range(8):
                for j in range(8):
                    feats[(str((i, j)),c)] = datum.grid[i][j]
                    featureSum += datum.grid[i][j]

            feats['slack',c] = MaxEntFeatureExtractor.V_FOR_SLACK -featureSum
        feats.divideAll(self.V_FOR_SLACK/Classfiers.MAXENT_V_CONST)
        return feats

class MaxEntTertiaryFeatureExtractor(FeatureExtractor):
    """
    Features valued 0,1,2 for maxEnt for each 8x8 grid point.
    The cutoff for 0,1,2 are 0, 11, 16 (all intensity are 0-16).
    """

    # Sum of features for a given (datum, label)
    SUM_OF_FEATURES = 64 * 2

    def getFeatures(self, datum):
        """
          Dictionary includes 64 features that marks the intensity on
          each grid space of the datum as a single feature
        """
        feats = util.Counter()
        for c in CLASSSET:
            featureSum = 0
            for i in range(8):
                for j in range(8):
                    intensity = datum.grid[i][j]
                    if intensity < 6:
                        brightness = 0
                    elif intensity < 11:
                        brightness = 1
                    else:
                        brightness = 2
                    feats[(str((i, j)), c)] = brightness
                    featureSum += feats[(str((i, j)),c)]

            feats['slack', c] = self.SUM_OF_FEATURES - featureSum
        feats.divideAll(self.SUM_OF_FEATURES / Classfiers.MAXENT_V_CONST)
        return feats

class DecisionTreeFeatureExtractor(FeatureExtractor):
    """
    Features valued 0,1,2 for DecisionTree for each 8x8 grid point.
    The cutoff for 0,1,2 are 0, 11, 16 (all intensity are 0-16).
    """
    def getFeatures(self, datum):
        """
        :param datum:
        :return: return the features of the brightness of each grid intentensity for the given datum.
        """
        feats = util.Counter()
        for i in range(8):
            for j in range(8):
                intensity = datum.grid[i][j]
                if intensity < 6:
                    brightness = 0
                elif intensity <11:
                    brightness = 1
                else:
                    brightness = 2
                feats[str((i, j))] = brightness

        return feats

    def getFeatureValues(self):
        return [0,1,2]

