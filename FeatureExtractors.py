import util

class FeatureExtractor:
    def getFeatures(self, datum):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class SymmetryExtractor(FeatureExtractor):
    def getFeatures(self, datum):
        """
          Dictionary includes a single feature that
          is the datum. This feature doesn't
          permit generalization.
        """
        feats = util.Counter()
        LeftTop = [][]
        for i in range(3):
            for j in range(3):


        feats['bias'] = 1.0
        return feats


