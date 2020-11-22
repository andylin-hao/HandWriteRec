from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import SGD, RMSprop
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import os


class Model:
    def __init__(self, train_percent=0.95):
        self.model = Sequential()

        self.train_percent = train_percent
        self.epoch = 40

        self.train_data = np.array(list())
        self.train_labels = np.array(list())
        self.validation_data = np.array(list())
        self.validation_labels = np.array(list())
        self.test_data = np.array(list())

        self.load_process_data()
        self.__name__ = 'Base'

    def load_process_data(self):
        raw_data = pd.read_csv('./data/train.csv')
        img_data = raw_data.iloc[:, 1:].values.astype(np.float)
        np.multiply(img_data, 1. / 255.)
        img_data = np.reshape(img_data, (-1, 28, 28, 1))

        labels = raw_data.iloc[:, 0].values
        labels = to_categorical(labels, num_classes=10).astype(np.uint8)

        train_size = int(42000 * self.train_percent)
        self.train_data = img_data[:train_size]
        self.train_labels = labels[:train_size]
        self.validation_data = img_data[train_size:]
        self.validation_labels = labels[train_size:]

        self.test_data = pd.read_csv('./data/test.csv')
        self.test_data = self.test_data.values.astype(np.float)
        np.multiply(self.test_data, 1. / 255.)
        self.test_data = np.reshape(self.test_data, (-1, 28, 28, 1))

    def predict(self):
        self.load()
        img_id = list(range(1, len(self.test_data) + 1))
        prediction = self.model.predict(self.test_data)
        prediction = [np.argmax(pred) for pred in prediction]
        data_frame = pd.DataFrame({
            'ImageId': img_id,
            'Label': prediction
        })
        data_frame.to_csv('result_' + self.__name__ + '.csv', index=False)

    def train(self, data_augmentation=False, save=True):
        callback = ReduceLROnPlateau(monitor='val_accuracy', patience=5, verbose=1, factor=0.5, min_lr=1e-6)
        if data_augmentation and self.__name__ == 'CNN':
            data_generate = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_epsilon=1e-6,
                zca_whitening=False,
                rotation_range=0,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.,
                zoom_range=0.,
                channel_shift_range=0.,
                fill_mode='nearest',
                cval=0.,
                horizontal_flip=True,
                vertical_flip=False,
                rescale=None,
                preprocessing_function=None,
                data_format=None,
                validation_split=0.0
            )
            data_generate.fit(self.train_data)
            self.model.fit(data_generate.flow(self.train_data, self.train_labels, 500), batch_size=500,
                           epochs=self.epoch, verbose=2,
                           validation_data=(self.validation_data, self.validation_labels), callbacks=[callback])
        else:
            self.model.fit(self.train_data, self.train_labels, batch_size=500, epochs=self.epoch, verbose=2,
                           validation_data=(self.validation_data, self.validation_labels), callbacks=[callback])
        scores = self.model.evaluate(self.validation_data, self.validation_labels)
        print(scores)

        if save:
            self.model.save(self.__name__ + '.h5', save_format='h5')

    def load(self):
        if not os.path.exists(self.__name__ + '.h5'):
            self.train()
        self.model = load_model(self.__name__ + '.h5')

    def plot(self):
        plot_model(self.model, self.__name__ + '.png', show_shapes=True)


class CNNModel(Model):
    def __init__(self, train_percent=0.95):
        super().__init__(train_percent)
        self.model.add(
            Conv2D(filters=32, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
        self.model.add(
            Conv2D(filters=32, kernel_size=(5, 5), padding='same', input_shape=(28, 28, 1), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(84, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])

        self.__name__ = 'CNN'


class LSTModel(Model):
    def __init__(self, train_percent=0.95):
        super().__init__(train_percent)
        self.model.add(LSTM(units=512, batch_input_shape=(None, 28, 28), unroll=True))
        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))
        sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

        self.train_data = self.train_data.reshape((-1, 28, 28))
        self.validation_data = self.validation_data.reshape((-1, 28, 28))
        self.test_data = self.test_data.reshape((-1, 28, 28))

        self.__name__ = 'LSTM'


class MLPModel(Model):
    def __init__(self, train_percent=0.95):
        super().__init__(train_percent)
        self.model.add(Dense(512, activation='relu', input_shape=(784,)))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.25))
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.train_data = self.train_data.reshape((-1, 784))
        self.validation_data = self.validation_data.reshape((-1, 784))
        self.test_data = self.test_data.reshape((-1, 784))

        self.__name__ = 'MLP'
