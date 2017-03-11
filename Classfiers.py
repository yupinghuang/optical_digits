import util
import FeatureExtractors
from matplotlib import pyplot as plt
import numpy as np
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
        self.weights = util.Counter()
        empiricalFeats = util.Counter()
        modelFeats = util.Counter()
        # TODO store feature vectors for all data
        # compute empirical value for each feature
        for datum in trainingSet:
            allLabelFeatures = self.featureExtractor.getFeatures(datum)
            for featureKey, value in allLabelFeatures.items():
                featureName, label = featureKey
                if label == datum.label:
                    empiricalFeats[featureName] += value
        empiricalFeats.divideAll(len(trainingSet))
        print 'emp', empiricalFeats
        # Intialize model feature values and the compute them through iteration
        for key in empiricalFeats.keys():
            modelFeats[key] = 0.

        # iteration
        # Either converge or just output the result after 10,000 iterations
        for i in xrange(10000):
            print 'iteration', i

            # compute expectation for each feature
            for datum in trainingSet:
                datumFeature = self.featureExtractor.getFeatures(datum)
                classificationDist = self.classificationDist(datumFeature)
                for label in CLASSSET:
                    # print 'class prob', classificationProb
                    for key in modelFeats.keys():
                        featValue = datumFeature[(key, label)]
                        modelFeats[key] += classificationDist[label] * featValue
            print 'all data done'
            modelFeats.divideAll(len(trainingSet))
            import numpy as np
            print 'model', modelFeats

            updateRatioVector = util.Counter()
            for featureName in empiricalFeats:
                if modelFeats[featureName] != 0.:
                    updateRatioVector[featureName] = (empiricalFeats[featureName]/modelFeats[featureName])

            for key, value in updateRatioVector.items():
                for label in CLASSSET:
                    oldWeight = self.weights.setdefault((key, label), 1.)
                    self.weights[(key, label)] = value * oldWeight
            print updateRatioVector[max(updateRatioVector)], updateRatioVector[min(updateRatioVector)]
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
            print exponents[key[1]]
        print 'exp', exponents
        dist = util.Counter()
        for key, value in exponents.items():
            dist[key] = math.exp(value)
        dist.normalize()
        print 'dist', dist
        return dist

    def predict(self, datumFeature):
        maxlabel = max(CLASSSET, key=lambda label: self.classificationDist(datumFeature)[label])
        # print maxlabel
        return maxlabel