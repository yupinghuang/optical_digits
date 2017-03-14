import util
class DecisionTreeNode(object):
    def __init__(self, unsplitFeatureList, feature=None, value=None, parent=None):
        """
        Initialize a decision tree node. This implementation always splits on
        feature values.

        :param unsplitFeatureList: the list of features that have not been split.
        :param feature: the name of the feature this node is split on.
        :param value: the value of the feature this node is split on.
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
        :return:
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
        if self.children:
            raise Exception("Cannot split on a node that has children")

        if featureName == '(5, 5)':
            print 'STOP HERE'
        nextFeatureList = list(self.unsplitFeatureList)
        nextFeatureList.remove(featureName)
        for dataPoint in self.featureData:
            featureValue = dataPoint[0][featureName]
            if featureValue == 0:
                print featureValue
            elif featureValue == 1:
                print featureValue
            elif featureValue == 2:
                print featureValue
            else:
                raise Exception
            childNode = self.children.setdefault(featureValue, DecisionTreeNode(unsplitFeatureList=nextFeatureList,
                feature=featureName, value=featureValue, parent=self))
            childNode.featureData.append(dataPoint)

    def find(self, featureVector):
        """
        Find a leaf node that the featureVector belongs
        :param featureVector:
        :return:
        """
        if not self.children:
            return self
        else:
            rightChild = self.children[featureVector[self.feature]]
            rightNode = rightChild.find(featureVector)
            return rightNode
        
    def getLabelDist(self):
        dist = util.Counter()
        for data, label in self.featureData:
            dist[label] += 1.
        dist.normalize()
        return dist

    def getmostProbableLabel(self):
        dist = self.getLabelDist()
        return dist.argMax()