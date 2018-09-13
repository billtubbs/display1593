# python
from platform import python_version
print('python: %s' % python_version())
# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# numpy
import numpy
print('numpy: %s' % numpy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas
print('pandas: %s' % pandas.__version__)
# statsmodels
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)
# theano
try:
    import theano
except ImportError:
    print('theano: not installed')
else:
    print('theano: %s' % theano.__version__)
# tensorflow
try:
    import tensorflow
except ImportError:
    print('tensorflow: not installed')
else:
    print('tensorflow: %s' % tensorflow.__version__)
# keras
try:
    import keras
except ImportError:
    print('keras: not installed')
else:
    print('keras: %s' % keras.__version__)
# pygame
try:
    import pygame
except ImportError:
    print('pygame: not installed')
else:
    print('pygame: %s' % pygame.__version__)