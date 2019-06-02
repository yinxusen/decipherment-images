"""
Copyright 2019 Xusen Yin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from keras import backend as K
from keras.initializers import RandomNormal, TruncatedNormal
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Lambda
from keras.models import Model, Sequential
from keras.regularizers import l2

from snn.kerasnn import KerasNNBase
from snn.utils import eprint


class SNN(KerasNNBase):
    def __init__(self,
                 model_path,
                 loss='binary_crossentropy',
                 optimizer='adadelta',
                 metrics=('mse',),
                 n_epoch=10000,
                 batch_size=128,
                 early_stop_patience=20,
                 init_epoch=0,
                 n_val=100):
        super(SNN, self).__init__(model_path=model_path,
                                  loss=loss,
                                  optimizer=optimizer,
                                  metrics=metrics,
                                  n_epoch=n_epoch,
                                  batch_size=batch_size,
                                  early_stop_patience=early_stop_patience,
                                  init_epoch=init_epoch)
        self.encoder = None
        self.n_val = n_val
        self.filter_W_init = TruncatedNormal(mean=0, stddev=1e-2)
        self.b_init = TruncatedNormal(mean=0.5, stddev=1e-2)
        self.fully_W_init = TruncatedNormal(mean=0, stddev=1e-1)

    def _build_model_impl(self, input_shape):
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        self.encoder = Sequential()
        self.encoder.add(Conv2D(64, (10, 10), activation='relu',
                                input_shape=input_shape,
                                kernel_regularizer=l2(2e-4),
                                kernel_initializer=self.filter_W_init,
                                bias_initializer=self.b_init))
        self.encoder.add(MaxPooling2D())
        self.encoder.add(Conv2D(128, (7, 7), activation='relu',
                                kernel_regularizer=l2(2e-4),
                                kernel_initializer=self.filter_W_init,
                                bias_initializer=self.b_init))
        self.encoder.add(MaxPooling2D())
        self.encoder.add(Conv2D(128, (4, 4), activation='relu',
                                kernel_regularizer=l2(2e-4),
                                kernel_initializer=self.filter_W_init,
                                bias_initializer=self.b_init))
        self.encoder.add(MaxPooling2D())
        self.encoder.add(Conv2D(256, (4, 4), activation='relu',
                                kernel_regularizer=l2(2e-4),
                                kernel_initializer=self.filter_W_init,
                                bias_initializer=self.b_init))
        self.encoder.add(Flatten())
        self.encoder.add(Dense(4096, activation="sigmoid",
                               kernel_regularizer=l2(1e-3),
                               kernel_initializer=self.fully_W_init,
                               bias_initializer=self.b_init))

        # encode each of the two inputs into a vector with the convnet
        encoded_l = self.encoder(left_input)
        encoded_r = self.encoder(right_input)

        # merge two encoded inputs with the l1 distance between them
        both = Lambda(lambda x: K.abs(x[0] - x[1]),
                      output_shape=lambda x: x[0])([encoded_l, encoded_r])
        prediction = Dense(1, activation='sigmoid',
                           bias_initializer=self.b_init,
                           kernel_initializer=self.fully_W_init)(both)
        self.model = Model(inputs=[left_input, right_input],
                           outputs=prediction)

        eprint(
            'Model built with {} parameters'.format(self.model.count_params()))

    def setup_fine_tuning(self):
        eprint('set model to fine tune mode ...')
        for layer in self.model.layers[:-2]:
            layer.trainable = False

    def _prepare_features(self, data):
        return data

    def _get_input_shape(self, generator):
        return generator.input_shape()

    def _fit_impl(self, generator):
        """
        fit a generator
        :param generator: an instance of SNNGenerator or its subclasses
        """
        steps_per_epoch = generator.data_size() // self.batch_size
        validation_steps = self.n_val // self.batch_size
        eprint('steps / epoch: {}'.format(steps_per_epoch))
        eprint('validation steps: {}'.format(validation_steps))

        self.model.fit_generator(
            generator=generator.train_generator(self.batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=self.n_epoch,
            validation_data=generator.dev_generator(self.batch_size),
            validation_steps=validation_steps,
            callbacks=self.callbacks,
            verbose=2,
            initial_epoch=self.init_epoch)

    def transform(self, images):
        raise NotImplementedError()

    def _save_impl(self, path):
        self._save_model(path, 'encoder', self.encoder)

    def evaluate(self, images):
        raise NotImplementedError()


class ReducedSNN(SNN):
    """
    Reduced version of SNN for smaller input shape, say, (35*35).
    """
    def _build_model_impl(self, input_shape):
        filter_W_init = RandomNormal(mean=0, stddev=1e-2)
        b_init = RandomNormal(mean=0.5, stddev=1e-2)
        fully_W_init = RandomNormal(mean=0, stddev=1e-1)

        left_input = Input(input_shape)
        right_input = Input(input_shape)

        self.encoder = Sequential()
        self.encoder.add(Conv2D(64, (4, 4), activation='relu',
                                input_shape=input_shape,
                                kernel_regularizer=l2(2e-4),
                                kernel_initializer=filter_W_init,
                                bias_initializer=b_init))
        self.encoder.add(MaxPooling2D())
        self.encoder.add(Conv2D(128, (3, 3), activation='relu',
                                kernel_regularizer=l2(2e-4),
                                kernel_initializer=filter_W_init,
                                bias_initializer=b_init))
        self.encoder.add(MaxPooling2D())
        self.encoder.add(Conv2D(128, (2, 2), activation='relu',
                                kernel_regularizer=l2(2e-4),
                                kernel_initializer=filter_W_init,
                                bias_initializer=b_init))
        self.encoder.add(MaxPooling2D())
        self.encoder.add(Conv2D(256, (2, 2), activation='relu',
                                kernel_regularizer=l2(2e-4),
                                kernel_initializer=filter_W_init,
                                bias_initializer=b_init))
        self.encoder.add(Flatten())
        self.encoder.add(Dense(512, activation="sigmoid",
                               kernel_regularizer=l2(1e-3),
                               kernel_initializer=fully_W_init,
                               bias_initializer=b_init))

        # encode each of the two inputs into a vector with the convnet
        encoded_l = self.encoder(left_input)
        encoded_r = self.encoder(right_input)

        # merge two encoded inputs with the l1 distance between them
        both = Lambda(lambda x: K.abs(x[0] - x[1]),
                      output_shape=lambda x: x[0])([encoded_l, encoded_r])
        prediction = Dense(1, activation='sigmoid',
                           bias_initializer=b_init,
                           kernel_initializer=fully_W_init)(both)
        self.model = Model(inputs=[left_input, right_input],
                           outputs=prediction)

        eprint(
            'Model built with {} parameters'.format(self.model.count_params()))


class L2SNN(KerasNNBase):
    def __init__(self,
                 model_path,
                 loss='binary_crossentropy',
                 optimizer='adadelta',
                 metrics=('mse',),
                 n_epoch=10000,
                 batch_size=128,
                 early_stop_patience=20,
                 init_epoch=0,
                 n_val=100):
        super(L2SNN, self).__init__(
            model_path=model_path,
            loss=loss,
            optimizer=optimizer,
            metrics=metrics,
            n_epoch=n_epoch,
            batch_size=batch_size,
            early_stop_patience=early_stop_patience,
            init_epoch=init_epoch)
        self.encoder = None
        self.n_val = n_val
        self.filter_W_init = RandomNormal(mean=0, stddev=1e-2)
        self.b_init = RandomNormal(mean=0.5, stddev=1e-2)
        self.fully_W_init = RandomNormal(mean=0, stddev=1e-1)

    def _build_model_impl(self, input_shape):
        left_input = Input(input_shape)
        right_input = Input(input_shape)

        self.encoder = Sequential()
        self.encoder.add(Conv2D(64, (10, 10), activation='relu',
                                input_shape=input_shape,
                                kernel_regularizer=l2(2e-4),
                                kernel_initializer=self.filter_W_init,
                                bias_initializer=self.b_init))
        self.encoder.add(MaxPooling2D())
        self.encoder.add(Conv2D(128, (7, 7), activation='relu',
                                kernel_regularizer=l2(2e-4),
                                kernel_initializer=self.filter_W_init,
                                bias_initializer=self.b_init))
        self.encoder.add(MaxPooling2D())
        self.encoder.add(Conv2D(128, (4, 4), activation='relu',
                                kernel_regularizer=l2(2e-4),
                                kernel_initializer=self.filter_W_init,
                                bias_initializer=self.b_init))
        self.encoder.add(MaxPooling2D())
        self.encoder.add(Conv2D(256, (4, 4), activation='relu',
                                kernel_regularizer=l2(2e-4),
                                kernel_initializer=self.filter_W_init,
                                bias_initializer=self.b_init))
        self.encoder.add(Flatten())
        self.encoder.add(Dense(4096, activation="sigmoid",
                               kernel_regularizer=l2(1e-3),
                               kernel_initializer=self.fully_W_init,
                               bias_initializer=self.b_init))

        # encode each of the two inputs into a vector with the convnet
        encoded_l = self.encoder(left_input)
        encoded_r = self.encoder(right_input)

        # merge two encoded inputs with the l1 distance between them
        both = Lambda(lambda x: K.square(x[0] - x[1]),
                      output_shape=lambda x: x[0])([encoded_l, encoded_r])
        prediction = Dense(1, activation='sigmoid',
                           bias_initializer=self.b_init,
                           kernel_initializer=self.fully_W_init)(both)
        self.model = Model(inputs=[left_input, right_input],
                           outputs=prediction)

        eprint(
            'Model built with {} parameters'.format(self.model.count_params()))

    def setup_fine_tuning(self):
        eprint('set model to fine tune mode ...')
        for layer in self.model.layers[:-2]:
            layer.trainable = False

    def _prepare_features(self, data):
        return data

    def _get_input_shape(self, generator):
        return generator.input_shape()

    def _fit_impl(self, generator):
        """
        fit a generator
        :param generator: an instance of SNNGenerator or its subclasses
        """
        steps_per_epoch = generator.data_size() // self.batch_size
        validation_steps = self.n_val // self.batch_size
        eprint('steps / epoch: {}'.format(steps_per_epoch))
        eprint('validation steps: {}'.format(validation_steps))

        self.model.fit_generator(
            generator=generator.train_generator(self.batch_size),
            steps_per_epoch=steps_per_epoch,
            epochs=self.n_epoch,
            validation_data=generator.dev_generator(self.batch_size),
            validation_steps=validation_steps,
            callbacks=self.callbacks,
            verbose=2,
            initial_epoch=self.init_epoch)

    def transform(self, images):
        raise NotImplementedError()

    def _save_impl(self, path):
        self._save_model(path, 'encoder', self.encoder)

    def evaluate(self, images):
        raise NotImplementedError()

