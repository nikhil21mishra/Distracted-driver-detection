import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import layers
from keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, 
                          Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D)
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Model, load_model, save_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

%matplotlib inline

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

from sklearn.model_selection import StratifiedKFold, cross_validate, LeaveOneGroupOut

from PIL import Image