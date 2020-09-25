import os
import argparse
import numpy as np
import pandas as pd
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, TensorBoard
from utils import INPUT_SHAPE, batch_generator
from sklearn.model_selection import train_test_split 
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten

np.random.seed(0)

# load the csv file and return numpy arrays
def load_data():
    data_df = pd.read_csv('data\\training_data_balanced.csv', names=['center', 'steering', 'throttle'])    
    X = data_df['center'].values
    y = data_df['steering'].values
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_valid, y_train, y_valid


# define the model architecture
def build_model():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()
    return model


# training pipeline
def train_model(model, X_train, X_valid, y_train, y_valid):
    
    # initialise hyperparameters
    batch_size = 42
    samples_per_epoch = 25788
    n_epoch = 50
    learning_rate = 1.0e-4

    # checkpoint to save the best model 
    # based on the validation accuracy
    checkpoint1 = ModelCheckpoint('model-{epoch:03d}-{val_loss:2f}.h5',
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only='true',
                                 mode='auto')

    # checkpoint to visualize the data in tensorboard
    checkpoint2 = TensorBoard(log_dir = './logs/highway-objdect-100k',
                             histogram_freq = 0,
                             write_graph = False,
                             write_images = False)

    # specify the cost function and learning algorithm
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=learning_rate))

    # start training using a batch generator
    model.fit_generator(batch_generator(X_train, y_train, batch_size, True),
                        samples_per_epoch,
                        n_epoch,
                        max_q_size=1,
                        validation_data=batch_generator(X_valid, y_valid, batch_size, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint1, checkpoint2],
                        verbose=1)


def main():
    data = load_data()
    model = build_model()
    train_model(model, *data)


if __name__ == '__main__':
    main()