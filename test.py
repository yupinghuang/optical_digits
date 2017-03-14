from DataSet import DataSet
import Classfiers
from FeatureExtractors import *
import argparse
import random

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Classifier tester.')
    parser.add_argument('--classifier',
                        help='The classifier to use: MIRA, MaxEnt, DT', type=str)
    args = parser.parse_args()

    trainingSet = DataSet('data/optdigits.tra')
    holdoutSet = DataSet('data/optdigits.hol')
    testSet = DataSet('data/optdigits.tes')
    def testMIRA():
        MIRAClassfier = Classfiers.MIRA(featureExtractor=AllGridExtractor())
        MIRAClassfier.train(trainingSet)
        print 'test on TRAINING set'
        MIRAClassfier.test(trainingSet)
        print 'test on HOLDOUT set'
        MIRAClassfier.test(holdoutSet)


    def testMaxEnt():
        maxEntClassifier = Classfiers.MaxEnt(featureExtractor=MaxEntBinaryFeatureExtractor())
        maxEntClassifier.train(trainingSet, iterations=15)
        print 'test on TRAINING set'
        maxEntClassifier.test(trainingSet)
        print 'test on HOLDOUT set'
        maxEntClassifier.test(holdoutSet)


    def testDT():
        dt = Classfiers.DecisionTree(featureExtractor=DecisionTreeFeatureExtractor())
        dt.train(trainingSet)
        print 'test on TRAINING set'
        dt.test(trainingSet)
        print 'test on HOLDOUT set'
        dt.test(holdoutSet)

    mapToClassifier = {'MIRA': testMIRA,
                       'MaxEnt': testMaxEnt,
                       'DT': testDT}


    toRun = mapToClassifier[args.classifier]
    toRun()

