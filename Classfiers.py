import util
import FeatureExtractors
from matplotlib import pyplot as plt
import numpy as np
from DataSet import CLASSSET
from DecisionTreeNode import DecisionTreeNode
import math

MIRACONST = 1.

class Classifer:
    def __init__(self, featureExtractor):
        self.weights = {}
        self.featureExtractor = featureExtractor

    def train(self, trainingSet):
        util.raiseNotDefined()

    def predict(self, datum):
        util.raiseNotDefined()

    def test(self, testingSet):
        rightPredicts = 0
        for datum in testingSet:
            datumFeature = self.featureExtractor.getFeatures(datum)
            classifiedLabel = self.predict(datumFeature)
            if classifiedLabel == datum.label:
                rightPredicts += 1
            # print "RIGHT LABEL", datum.label
            # print "CLASSIFIED LABEL", classifiedLabel
            # plot = datum.draw()
            # plt.show(plot)

        print "RIGHTLY PREDICTED:", float(rightPredicts)/len(testingSet)

class MIRA(Classifer):
    def __init__(self, featureExtractor, miraConst = MIRACONST):
        self.miraConst = miraConst
        Classifer.__init__(self, featureExtractor)

    def train(self, trainingSet):
        #TODO: add iteration part here

        for label in CLASSSET:
            self.weights[label] = util.Counter()

        for datum in trainingSet:
            datumFeature = self.featureExtractor.getFeatures(datum)

            classifiedLabel = self.predict(datumFeature)
            # print "RIGHT LABEL", datum.label

            if datum.label!= classifiedLabel:
                tau = ((self.weights[classifiedLabel] - self.weights[datum.label])*datumFeature + 1)/(2.0*(datumFeature*datumFeature))
                tau = min(tau, self.miraConst)

                datumFeatureMultiplied = datumFeature.copy()
                datumFeatureMultiplied.multiplyAll(tau)
                self.weights[classifiedLabel] -= datumFeatureMultiplied
                self.weights[datum.label] += datumFeatureMultiplied
                # print self.weights[datum.label]

    def predict(self, datumFeature):
        maxlabel = max(CLASSSET, key=lambda label: self.weights[label] * datumFeature)
        # print maxlabel
        return maxlabel


class MaxEnt(Classifer):
    def train(self, trainingSet):
        self.weights = util.Counter()
        empiricalFeats = util.Counter()
        modelFeats = util.Counter()
        # store feature vectors for all data
        trainingSetFeatures = {}
        for datum in trainingSet:
            trainingSetFeatures[datum] = self.featureExtractor.getFeatures(datum)

        # compute empirical value for each feature
        for datum in trainingSet:
            for featureKey, value in trainingSetFeatures[datum].items():
                    if featureKey[1] == datum.label:
                        empiricalFeats[featureKey] += value
        empiricalFeats.divideAll(len(trainingSet))

        # Intialize model feature values and the compute them through iteration
        for key in empiricalFeats.keys():
            modelFeats[key] = 0.

        # iteration
        # Either converge or just output the result after 10,000 iterations

        for i in xrange(10000):
            print 'iteration', i

            # compute expectation for each feature
            for datum in trainingSet:
                datumFeature = trainingSetFeatures[datum]
                classificationDist = self.classificationDist(datumFeature)

                for key in datumFeature:
                    featValue = datumFeature[key]
                    modelFeats[key] += classificationDist[key[1]] * datumFeature[key]

            print 'all data done'
            modelFeats.divideAll(len(trainingSet))

            updateRatioVector = util.Counter()
            for featureKey, value in empiricalFeats.items():
                if value != 0.:
                    if modelFeats[featureKey] != 0.:
                        updateRatioVector[featureKey] = empiricalFeats[featureKey]/modelFeats[featureKey]
                    else:
                        updateRatioVector[featureKey] = 1.0


            for key, value in updateRatioVector.items():
                oldWeight = self.weights.setdefault(key, 1.)
                self.weights[key] = value * oldWeight

            # check for convergence
            converge = True
            for key, value in updateRatioVector.items():
                if abs(value-1.) > 1e-2:
                    converge = False
                    break

            if converge:
                print 'maxent converged!'
                break

    def classificationDist(self, feats):
        exponents = util.Counter()
        for key, value in feats.items():
            exponents[key[1]] += self.weights.setdefault(key, 1.) * value
        dist = util.Counter()
        overflowCount = 0
        for key, value in exponents.items():
            try:
                dist[key] = math.exp(value)
            except OverflowError:
                if overflowCount==1:
                    raise "Two labels have overflown likelihood"
                overflowCount += 1
                print 'likelihood overflown for label', key
                dist[key] = float("inf")
        dist.normalize()
        return dist

    def predict(self, datumFeature):
        maxlabel = max(CLASSSET, key=lambda label: self.classificationDist(datumFeature)[label])
        # print maxlabel
        return maxlabel

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
        featureToSplit = min(remainingFeatures, key=lambda f: self.getConditionalEntropy(f, remainingData,
                            self.featureExtractor.getFeatureValues()))
        return featureToSplit

    def getConditionalEntropy(self,feature, trainingSetFeatures, featureValues):
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
        featureValueCount = np.sum(probabilityMatrix,axis = 1)
        featureValueProb = featureValueCount/len(trainingSetFeatures)

        #calculate out conditional entropy
        condEntropy = 0
        for featureValue in featureValues:
            sumFeatureValueEntropy = 0
            for label in CLASSSET:
                numerator = probabilityMatrix[featureValue][int(label)]
                denomintor = featureValueCount[featureValue]
                if numerator == 0 or denomintor== 0:
                    continue
                else:
                    condProb = numerator / denomintor
                    sumFeatureValueEntropy += condProb * math.log(condProb, 2)
            condEntropy -= featureValueProb[featureValue] * sumFeatureValueEntropy

        return condEntropy

    def predict(self, features):
        leafNode = self.root.find(features)
        return leafNode.getmostProbableLabel()
