from DataSet import DataSet
import Classfiers
from FeatureExtractors import AllGridExtractor
import random

if __name__=='__main__':
    # Read the data and randomly plot 10 of them
    trainingSet = DataSet('data/optdigits.tra')
    holdoutSet = DataSet('data/optdigits.hol')
    testSet = DataSet('data/optdigits.tes')
    # for i in range(10):
    #     trainingSet.plotRandom()

    # trainingSubset = [random.choice(trainingSet) for i in range(10)]
    # testingSubset = [testingSet[i] for i in range(10)]
    # MIRAClassfier.train(trainingSubset)
    # MIRAClassfier.test(testingSet)
    MIRAClassfier = Classfiers.MIRA(featureExtractor=AllGridExtractor())
    MIRAClassfier.train(trainingSet)
    MIRAClassfier.test(holdoutSet)

