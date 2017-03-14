import util
class DecisionTreeNode(object):
    def __init__(self, unsplitFeatureList, feature=None, value=None, parent=None):
        """
        Initialize a decision tree node. This implementation always splits on
        feature values.

        :param unsplitFeatureList: the list of features that have not been split.
        :param feature: the name of the feature its children are split on.
        :param value: the value of the feature THIS node is split on.
        :param parent: the parent.
        """
        self.unsplitFeatureList = unsplitFeatureList
        self.parent = parent
        self.children = {}
        self.feature = feature
        self.value = value
        # featureData is a list of (featureVector, label) tuples
        self.featureData = []

    @classmethod
    def getRoot(cls, featureList, featureData):
        """
        get the root node of a decision tree for a dataset
        :param featureList: the list of the names of the features
        :param featureData: a list of tuples (featureVector, label) where featureVector is the output
        of FeatureExtractor for a given datum and label the datum's label.
        :return: the root node of a Decision Tree.
        """
        root = DecisionTreeNode(unsplitFeatureList=featureList)
        root.featureData = featureData
        return root

    def splitOnFeature(self, featureName):
        """
        Split the current node
        :param featureName: the name of the feature to split
        :return:
        """
        self.feature = featureName
        if self.children:
            raise Exception("Cannot split on a node that has children.")

        nextFeatureList = list(self.unsplitFeatureList)
        nextFeatureList.remove(featureName)
        for dataPoint in self.featureData:
            featureValue = dataPoint[0][featureName]
            childNode = self.children.setdefault(featureValue, DecisionTreeNode(unsplitFeatureList=nextFeatureList,
                feature=None, value=featureValue, parent=self))
            childNode.featureData.append(dataPoint)

    def getmostProbableLabel(self):
        """
        Get the most probable label (the majority) a tree node represents.
        :return: a label.
        """
        dist = self._getLabelDist()
        return dist.argMax()

    def _getLabelDist(self):
        """
        Get the distribution of labels for the current node.
        :return: A Counter object representing the normalized distribution.
        """
        dist = util.Counter()
        for data, label in self.featureData:
            dist[label] += 1.
        dist.normalize()
        return dist

    def find(self, featureVector):
        """
        Find a leaf node that the featureVector belongs when the tree predicts the class of a datum.
        :param featureVector: a featureVector which is the output of FeatureExtractor for a given datum.
        :return: the node that it belongs in the tree.
        """
        if not self.children:
            return self
        else:
            if featureVector[self.feature] not in self.children.keys():
                return self
            else:
                nextChild = self.children[featureVector[self.feature]]
                return nextChild.find(featureVector)

