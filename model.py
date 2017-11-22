from keras.models import Sequential
from keras.layers import Input, Dense, GRU, Lambda, Conv1D, MaxPooling1D, Concatenate, Multiply
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras.layers import Convolution2D, Activation, MaxPooling2D, Dropout, LSTM, Flatten, Merge, TimeDistributed
import numpy as np

# Generate fake data
# Assumed to be 1730 grayscale video frames
def construct_model(args):

    x1 = Input(shape=(None, args.mfb_dim), name='mfb_input')

    # pre-process mfb features (reduce temporal resolution)
    x1 = Conv1D(filters=args.nb_conv_filters,
                kernel_size=args.filter_length,
                padding='same',
                activation='tanh')(x1)
    x1 = MaxPooling1D(pool_size=3,strides=3)(x1)
    x1 = Dense(args.dense_layer_width, activation='relu')(x1)
    x1 = LSTM(args.lstm_width)
    x1 = LSTM(args.lstm_width)
    x1 = Dense(args.dense_layer_width, activation='relu')(x1)
    y = Dense(args.output_dim, activation='softmax')(x1)

    # create model
    my_model = Model(inputs=x1, outputs=y)

    return my_model
