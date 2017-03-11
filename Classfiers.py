import util
import FeatureExtractors
from matplotlib import pyplot as plt
from DataSet import CLASSSET
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
        #TODO: add iteration part here
        self.weights = util.Counter()
        # compute empirical value for each feature

        # TODO move empirical feature somewhere else
        empiricalFeats = util.Counter()
        modelFeats = util.Counter()
        '''
        for datum in trainingSet:
            for i in range(8):
                for j in range(8):
                    empiricalFeats[(i, j)] += datum.grid[i,j]
        '''
        for datum in trainingSet:
            allLabelFeatures = self.featureExtractor.getFeatures(datum)
            for featureKey, value in allLabelFeatures.items():
                featureName, label = featureKey
                if label == datum.label:
                    empiricalFeats[featureName] += value
        empiricalFeats.divideAll(len(trainingSet))
        for key in empiricalFeats.keys():
            modelFeats[key] = 0.

        # iteration
        # Either converge or just output the result after 10,000 iterations
        for i in xrange(10000):
            # compute expectation for each feature
            for datum in trainingSet:
                for label in CLASSSET:
                    for key in modelFeats.keys():
                        datumFeature = self.featureExtractor.getFeatures(datum)
                        classificationProb = self.classificationProb(datumFeature,label,datum)
                        featValue = datumFeature[(key, label)]
                        modelFeats[key] += classificationProb*featValue
            modelFeats.divideAll(len(trainingSet))

            updateRatioVector = (empiricalFeats/modelFeats)**(1/self.featureExtractor.V_FOR_SLACK)
            for key, value in self.weights.items():
                self.weights[key] = value * updateRatioVector[key[0]]
            if max(updateRatioVector) < 1e-2:
                print 'maxent converged!'
                break

    def classificationProb(self, feats, label):
        labelWeights = util.Counter()
        for key, value in self.weights.items():
            if key[1] == label:
                labelWeights[key] = value

        numerator = math.exp(labelWeights * feats)
        denominator = math.exp(self.weights * feats)
        return numerator/denominator

    #TODO: WRITE PREDICT
    def predict(self, datumFeature):
        maxlabel = max(CLASSSET, key=lambda label: self.classificationProb(datumFeature, label))
        # print maxlabel
        util.raiseNotDefined()