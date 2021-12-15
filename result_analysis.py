#!usr/bin/env python3

from matplotlib import pyplot
import pandas as pd

# load results
results = pd.read_csv('analysis.csv')

# plot accuracy
results.plot(x='model_name', y='accuracy', kind='bar')
pyplot.title('model accuracy')
pyplot.savefig('accuracy.png')
pyplot.show()

# plot nonguesses
results.plot(x='model_name', y='nonguesses', kind='bar')
pyplot.ylim(75, 85)
pyplot.title('model non-guesses')
pyplot.savefig('nonguesses.png')
pyplot.show()
