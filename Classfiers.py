import util
import FeatureExtractors
from matplotlib import pyplot as plt
import numpy as np
from DataSet import CLASSSET
import sys
import random
from DecisionTreeNode import DecisionTreeNode
import math
# The update rate constant for MIRA, it seems like in our case the optimal value is 0.05 for
# the AllGridExtractor
MIRACONST = 0.05
MAXENT_V_CONST = 4.

class Classifer:
    """
    The interface for all Classifier classes.
    """
    def __init__(self, featureExtractor):
        """
        Initializer.
        :param featureExtractor: the featureExtractor INSTANCE that should be used for the classifier.
        """
        self.weights = {}
        self.featureExtractor = featureExtractor

    def train(self, trainingSet):
        """
        Train the classifier.
        :param trainingSet:
        :return:
        """
        util.raiseNotDefined()

    def predict(self, datum):
        """
        Returns the predicted label of a datum object.
        :param datum:
        :return:
        """
        util.raiseNotDefined()

    def test(self, testingSet):
        """
        testing method for all Classifiers. Print out a bunch of helpful metrics.
        :param testingSet:
        :return:
        """
        rightPredicts = 0

        overalPredicts = 0
        wrongStats = util.Counter()
        totalCount = util.Counter()
        for datum in testingSet:
            datumFeature = self.featureExtractor.getFeatures(datum)
            classifiedLabel = self.predict(datumFeature)
            totalCount[datum.label] += 1
            if classifiedLabel == datum.label:
                rightPredicts += 1
            else:
                wrongStats[datum.label] += 1

        for key, value in totalCount.items():
            wrongStats[key] = wrongStats[key] / float(value)
        print 'fraction of instances incorrectly classified for each label', wrongStats
            # print "RIGHT LABEL", datum.label
            # print "CLASSIFIED LABEL", classifiedLabel
            # plot = datum.draw()
            # plt.show(plot)

        print "RIGHTLY PREDICTED:", float(rightPredicts)/len(testingSet)

class MIRA(Classifer):
    """
    The MIRA classifier
    """
    def __init__(self, featureExtractor, miraConst = MIRACONST):
        # self.miraConst is the upper bound for the tau ratio calculated during MIRA update.
        self.miraConst = miraConst
        Classifer.__init__(self, featureExtractor)

    def train(self, trainingSet):
        """
        Update self.weights given the trainingSet.
        :param trainingSet: A DataSet object representing the training set.
        :return:
        """
        for label in CLASSSET:
            self.weights[label] = util.Counter()

        for datum in trainingSet:
            datumFeature = self.featureExtractor.getFeatures(datum)
            classifiedLabel = self.predict(datumFeature)

            if datum.label!= classifiedLabel:
                # Wrong prediction, update weights
                tau = ((self.weights[classifiedLabel] - self.weights[datum.label])*datumFeature + 1)/(2.0*(datumFeature*datumFeature))
                print 'tau', tau
                tau = min(tau, self.miraConst)
                # need to copy the feature dictionary in order to do the multiplication and updates.
                datumFeatureMultiplied = datumFeature.copy()
                datumFeatureMultiplied.multiplyAll(tau)
                self.weights[classifiedLabel] -= datumFeatureMultiplied
                self.weights[datum.label] += datumFeatureMultiplied

    def predict(self, datumFeature):
        maxlabel = max(CLASSSET, key=lambda label: self.weights[label] * datumFeature)
        # print self.weights[maxlabel] * datumFeature
        return maxlabel


class MaxEnt(Classifer):
    """
    The maximum entropy classifier using generalized iterative scaling for training.
    """
    def train(self, trainingSet, iterations=10):
        """
        Update self.weights based on the training data. It apply the trained model to the training
        in each iteration so that a human can figure out the best number of iterations by looking at
        the output.
        :param trainingSet: A DataSet object representing the training set.
        :param iterations: Upper bound of number of iterations.
        :return:
        """
        self.weights = util.Counter()
        empiricalFeats = util.Counter()

        # store feature vectors for all data to speed up the process
        trainingSetFeatures = {}
        for datum in trainingSet:
            trainingSetFeatures[datum] = self.featureExtractor.getFeatures(datum)

        # compute empirical value for each feature
        for datum in trainingSet:
            for featureKey, value in trainingSetFeatures[datum].items():
                    if featureKey[1] == datum.label:
                        empiricalFeats[featureKey] += value
                    else:
                        # Just so the key exists.
                        empiricalFeats[featureKey] += 0.
        empiricalFeats.divideAll(len(trainingSet))
        # Test how well the uniform model fits the data as a baseline
        print 'RANDOM GUESS'
        self.test(trainingSet)

        # iterations
        # Either converge or just output the result after a certain number of iterations

        for i in xrange(iterations):
            print 'iteration', i
            modelFeats = util.Counter()
            # compute expectation for each feature under the model
            for datum in trainingSet:
                datumFeature = trainingSetFeatures[datum]
                classificationDist = self.classificationDist(datumFeature)

                for key, featValue in datumFeature.items():
                    modelFeats[key] += classificationDist[key[1]] * featValue
            modelFeats.divideAll(len(trainingSet))

            # compute update.
            updateRatioVector = util.Counter()
            for featureKey, empiricalFeat in empiricalFeats.items():
                if modelFeats[featureKey] != 0.:
                    updateRatioVector[featureKey] = empiricalFeat/modelFeats[featureKey]
                else:
                    updateRatioVector[featureKey] = 1.

            # Update assuming that the V parameter is scaled down to MAXENT_V_CONST.
            for key, updateRatio in updateRatioVector.items():
                oldWeight = self.weights.setdefault(key, 1.)
                self.weights[key] = updateRatio**(1./MAXENT_V_CONST) * oldWeight

            # Test how well the model fits the data
            testData = trainingSet
            self.test(testData)
            # check for convergence
            '''
            converge = True
            for key, value in updateRatioVector.items():
                if abs(value-1.) > 1e-2:
                    converge = False
                    break

            if converge:
                print 'maxent converged!'
                break
            '''
            averageUpdateRatioDev = np.average([abs(value-1.) for key, value in updateRatioVector.items() if value!=0.])
            print 'average update ratio absolutedeviation from 1.', averageUpdateRatioDev
            if averageUpdateRatioDev < 1e-2:
                print 'maxent converged'
                # break

    def classificationDist(self, feats):
        """
        Helper function that calculates the classification distribution given a feature vector.
        i.e. p(c|x).
        :param feats: the feature vector
        :return: The distribution represented by a Counter to which the label c is the key.
        """
        exponents = util.Counter()
        for key, value in feats.items():
            exponents[key[1]] += self.weights.setdefault(key, 1.) * value

        dist = util.Counter()

        for label in CLASSSET:
            labelExponent = exponents[label]
            denominator = 1.
            # print exponents
            for key,value in exponents.items():
                if key != label:
                    try:
                        denominator += math.exp(value - labelExponent)
                    except OverflowError:
                        # if the exponent is large, approximate by a very large value that does not cause
                        # nans
                        denominator = sys.float_info.max
            dist[label] = 1./denominator
        dist.normalize()
        return dist

    def predict(self, datumFeature):
        dist = self.classificationDist(datumFeature)
        return dist.argMax()

class DecisionTree(Classifer):
    def train(self, trainingSet):
        # store feature vectors for all data
        featureData = []
        featureList = []
        for datum in trainingSet:
            featureVector = self.featureExtractor.getFeatures(datum)
            featureData.append((featureVector, datum.label))
            if not featureList:
                featureList = featureVector.keys()
        self.root = DecisionTreeNode.getRoot(featureList, featureData)
        self.buildTree(self.root)

    def buildTree(self, root):
        """
        recursive method to build the decision tree.
        :param root:
        :return:
        """
        # Base case
        # if there is no more attribute to split on
        if len(root.unsplitFeatureList) == 0 or len(root.featureData)==1:
            return

        # if all the examples left are of the same attributes
        allTheSame = True
        for feature in root.unsplitFeatureList:
            if not allTheSame:
                break
            firstValue = root.featureData[0][0][feature]
            for data in root.featureData:
                if data[0][feature] != firstValue:
                    allTheSame = False
                    break
        if allTheSame:
            return

        # if all the example lefts are of the same label
        allTheSame = True
        firstLabel = root.featureData[0][1]
        for data in root.featureData:
            if data[1] != firstLabel:
                allTheSame = False
                break
        if allTheSame:
            return

        featureToSplit = self.pickFeatureToSplit(root.unsplitFeatureList, root.featureData)
        root.splitOnFeature(featureToSplit)
        for key,child in root.children.items():
            self.buildTree(child)

    def pickFeatureToSplit(self, remainingFeatures, remainingData):
        """
        Pick the next feature (the one with the lowest conditional entropy) to split.
        :param remainingFeatures: A list of the unsplit features.
        :param remainingData: The data to be split.
        :return:
        """
        featureToSplit = min(remainingFeatures, key=lambda f: self.getConditionalEntropy(f, remainingData,
                            self.featureExtractor.getFeatureValues()))
        return featureToSplit

    def getConditionalEntropy(self, feature, trainingSetFeatures, featureValues):
        """
        Calculate the conditional entropy of a set of training data given a feature.
        ASSUMING that feature values are 0,1,2,3...
        :param feature: Name of the feature to condition on.
        :param trainingSetFeatures: the training data.
        :param featureValues: The possible values for the feature.
        :return: the value of conditional entropy
        """
        # create a matrix to store the value
        probabilityMatrix = np.zeros((len(featureValues),len(CLASSSET)))
        # going through each datum to count their occurance for the label and feature
        # assume that feature values are 0,1,2,....
        # TODO: change into dictionary
        for datum in trainingSetFeatures:
            datumFeature = datum[0]
            featureValue = datumFeature[feature]
            label = datum[1]
            probabilityMatrix[featureValue][int(label)] += 1

        # sum to find the sum of
        featureValueCount = np.sum(probabilityMatrix, axis = 1)
        featureValueProb = featureValueCount/len(trainingSetFeatures)

        #calculate out conditional entropy
        condEntropy = 0.
        for featureValue in featureValues:
            sumFeatureValueEntropy = 0.
            for label in CLASSSET:
                numerator = probabilityMatrix[featureValue][int(label)]
                denomintor = featureValueCount[featureValue]
                if numerator == 0. or denomintor== 0.:
                    continue
                else:
                    condProb = numerator / denomintor
                    sumFeatureValueEntropy += condProb * math.log(condProb, 2)
            condEntropy -= featureValueProb[featureValue] * sumFeatureValueEntropy

        return condEntropy

    def predict(self, features):
        """
        Find the node in the decision tree that a given feature vector should end up in and return
        the majority label in that node.
        :param features:
        :return:
        """
        leafNode = self.root.find(features)
        return leafNode.getmostProbableLabel()
