import util
import FeatureExtractors
from matplotlib import pyplot as plt
from DataSet import CLASSSET

MIRACONST = 1.

class Classifer:
    def __init__(self):
        self.weights = {}

    def train(self, trainingSet):
        util.raiseNotDefined()

    def predict(self, datum):
        util.raiseNotDefined()

    def test(self, testingSet):
        util.raiseNotDefined()

class MIRA(Classifer):
    def train(self, trainingSet):
        #TODO: add iteration part here
        se = FeatureExtractors.SymmetryExtractor()
        for label in CLASSSET:
            self.weights[label] = util.Counter()

        for datum in trainingSet:
            datumFeature = se.getFeatures(datum)

            classifiedLabel = self.predict(datumFeature)
            # print "RIGHT LABEL", datum.label
            if datum.label!= classifiedLabel:
                tau = ((self.weights[classifiedLabel] - self.weights[datum.label])*datumFeature + 1)\
                      /(2.0*(datumFeature*datumFeature))
                tau = min(tau, MIRACONST)
                datumFeatureMultiplied = datumFeature.copy()
                datumFeatureMultiplied.multiplyAll(tau)
                self.weights[classifiedLabel] -= datumFeatureMultiplied
                self.weights[datum.label] += datumFeatureMultiplied
                print self.weights[datum.label]

    def predict(self, datumFeature):
        maxlabel = max(CLASSSET, key=lambda label: self.weights[label] * datumFeature)
        # print maxlabel
        return maxlabel

    def test(self, testingSet):
        rightPredicts = 0
        se = FeatureExtractors.SymmetryExtractor()
        for datum in testingSet:
            datumFeature = se.getFeatures(datum)
            classifiedLabel = self.predict(datumFeature)
            if classifiedLabel == datum.label:
                rightPredicts += 1

        print "RIGHTLY PREDICTED:", float(rightPredicts)/len(testingSet)
