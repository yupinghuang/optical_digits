from DataSet import DataSet

if __name__=='__main__':
    # Read the data and randomly plot 10 of them
    d = DataSet('data/optdigits.tra')
    for i in range(10):
        d.plotRandom()
