import util
import numpy as np

SYMMETRYTHRESHOLD = 0.2

class FeatureExtractor:
    def getFeatures(self, datum):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

    # TODO: delete this maybe
    def getFeatureNames(self):
        return ['bias', 'symmetry']


class SymmetryExtractor(FeatureExtractor):
    def getFeatures(self, datum):
        """
          Dictionary includes a single feature that
          is the datum. This feature doesn't
          permit generalization.

          get the four sections of the image and
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
        # print "leftColorIntensity", upperColorIntensity
        # print "rightColorIntensity", lowerColorIntensity

        ratio = lowerColorIntensity/float(upperColorIntensity)
        feats['asymmetry'] = ratio
        return feats


