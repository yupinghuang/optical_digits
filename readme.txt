The data is in the folder called ¡°data¡± in the code folder ¡°optical_digits¡±. Otherwise, the relevant data is downloadable from https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/

To run the project main function test.py, run following commands in folder ¡°optical_digits¡±, essentially ¡°python test.py¡± with arguments.

For MIRA, run command ¡°python test.py ¡ª-classifier MIRA¡° 

For Maximum entropy classifier with features in full values (0 to 16), run command ¡°python test.py ¡ª-classifier MaxEntFull¡°

For Maximum entropy classifier with features in tertiary values (discretized feature values), run command ¡°python test.py ¡ª-classifier MaxEntTertiary¡°

For Decision Tree classifier, run command ¡°python test.py ¡ª-classifier DT¡°

For help message, run command ¡°python test.py ¡ªh¡°
