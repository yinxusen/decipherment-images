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

import importlib
import json
from abc import abstractmethod, ABCMeta

from keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback, \
    CSVLogger
from keras.models import model_from_json

from snn.utils import eprint


class KerasNNBase(object):

    __metaclass__ = ABCMeta

    def __init__(self,
                 model_path,
                 loss='binary_crossentropy',
                 optimizer='adadelta',
                 metrics=('mse',),
                 n_epoch=10000,
                 batch_size=128,
                 early_stop_patience=20,
                 init_epoch=0):
        super(KerasNNBase, self).__init__()
        self.model = None
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = list(metrics)
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.patience = early_stop_patience
        self.callbacks = []
        self.init_epoch = init_epoch
        self.model_path = model_path
        assert self.model_path is not None, 'model path should be provided.'

    def _build_model(self, input_shape, force_rebuilt=False,
                     build_model_only=False):
        if self.model is None or force_rebuilt:
            eprint('build model from scratch')
            self._build_model_impl(input_shape)
            eprint('reset init epoch to zero')
            self.init_epoch = 0
        else:  # train on a existed model
            pass

        if not build_model_only:
            self._create_callbacks()
            self._compile_model()
            self._restore_from_ckpt()

    @abstractmethod
    def _build_model_impl(self, input_shape):
        raise NotImplementedError()

    @abstractmethod
    def _prepare_features(self, data):
        """default behavior is doing nothing"""
        return data

    def _compile_model(self):
        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                           metrics=self.metrics)

    def _create_callbacks(self):
        # add checkpoint
        checkpoint_path = '{}/checkpoint.h5'.format(self.model_path)
        log_path = '{}/training.log'.format(self.model_path)
        epoch_count_path = '{}/epoch_count.txt'.format(self.model_path)
        checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss',
                                     verbose=1, save_best_only=True, mode='min')
        self.callbacks.append(checkpoint)
        self.callbacks.append(EarlyStopping(patience=self.patience))
        restoring_weights_callback = LambdaCallback(
            on_train_end=lambda _: self.model.load_weights(checkpoint_path))
        self.callbacks.append(restoring_weights_callback)

        def save_epoch(f, epoch):
            """the reason of epoch + 2:
            epoch in the callback is real epoch - 1
            """
            with open(f, 'w') as out:
                out.write('{}\n'.format(epoch + 2))
                out.flush()

        epoch_count_callback = LambdaCallback(
            on_epoch_end=lambda epoch, _: save_epoch(epoch_count_path, epoch))
        self.callbacks.append(epoch_count_callback)
        self.callbacks.append(CSVLogger(log_path, append=True))

    def _restore_from_ckpt(self):
        assert self.model is not None
        try:
            eprint('attempt to restore weights from the checkpoint ...')
            self.model.load_weights('{}/checkpoint.h5'.format(self.model_path))
        except Exception as e:
            eprint('restore from checkpoint failed: {}'.format(e))
            try:
                eprint('attempt to restore weights from saved model ...')
                self.model.load_weights('{}/model.h5'.format(self.model_path))
            except Exception as e:
                eprint('restore from saved model failed: {}'.format(e))
        try:
            eprint('attempt to restore epoch count ...')
            with open('{}/epoch_count.txt'.format(self.model_path), 'r') as f:
                epoch_count = int(f.readline())
                eprint('reset init epoch from {} to {}'.
                              format(self.init_epoch, epoch_count))
                self.init_epoch = epoch_count
        except Exception as e:
            eprint('restore epoch count failed: {}'.format(e))

    @abstractmethod
    def _get_input_shape(self, data):
        raise NotImplementedError()

    def fit(self, data):
        self._build_model(self._get_input_shape(data))
        self._fit_impl(self._prepare_features(data))

    @abstractmethod
    def _fit_impl(self, data):
        raise NotImplementedError()

    @abstractmethod
    def transform(self, images):
        raise NotImplementedError()

    @classmethod
    def _save_model(cls, path, prefix, model):
        with open('{}/{}.json'.format(path, prefix), 'w') as f:
            f.write(model.to_json())
        model.save_weights('{}/{}.h5'.format(path, prefix), overwrite=True)

    @abstractmethod
    def _save_impl(self, path):
        eprint('caution: model save fall back into its base class,'
               ' extended objects would not be saved.')
        eprint('implement your own _saveImpl if you need more.')
        pass

    def save(self, path=None):
        if path is None:
            path = self.model_path
        assert path != "", 'blank path is not allowed.'
        self._save_model(path, 'model', self.model)
        with open('{}/meta.json'.format(path), 'wb') as meta:
            meta.write(json.dumps(
                {'mod': self.__module__, 'clazz': self.__class__.__name__}))
        self._save_impl(path)

    @classmethod
    def _load_model(cls, path, prefix):
        json_file = open('{}/{}.json'.format(path, prefix), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model.load_weights('{}/{}.h5'.format(path, prefix))
        return model

    @classmethod
    def _load_impl(cls, path, obj):
        """
        this method gives children classes the ability to load their own
        metadata.
        :param path: model_path
        :param obj: an object of the class instantiated in `load` with given
                    arguments.
        """
        return obj

    @classmethod
    def load(cls, path, init_args=None):
        """
        load model from model_path
        Note: this method doesn't load weights or structures from the given path
        All loading behaviors postpone to the calling of the `fit` method.
        So if you want to load then predict, use _loadModel instead of the
        method.
        Remember that do not load correlated models separately, because that
        beaks the connection between them. Always create them again
        (i.e. initializing a new object), then load weights only.
        :param path: model_path
        :param init_args: initial arguments for the class.
        """
        assert path != "", 'blank path is not allowed.'
        with open('{}/meta.json'.format(path), 'rb') as meta:
            metadata = json.load(meta)
        mod = importlib.import_module(metadata['mod'])
        clazz = getattr(mod, metadata['clazz'])
        if init_args is None:
            obj = clazz(model_path=path)
        elif isinstance(init_args, dict):
            if init_args.get('model_path') is None:
                path_arg = {'model_path': path}
                path_arg.update(init_args)
                obj = clazz(**path_arg)
            else:
                obj = clazz(**init_args)
        else:
            raise ValueError(
                'unknown init args for loading Keras model: {}'.format(
                    init_args))
        obj = cls._load_impl(path, obj)
        return obj

    @abstractmethod
    def evaluate(self, images):
        raise NotImplementedError()

