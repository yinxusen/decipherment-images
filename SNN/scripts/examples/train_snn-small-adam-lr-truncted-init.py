import sys

from keras import backend
from keras.optimizers import Adam

from snn.generators import OmniglotGenerator
from snn.oneshot import SNN
from snn.utils import eprint

if __name__ == '__main__':
    generator = OmniglotGenerator(sys.argv[1])  # load training data from dir
    model_path = sys.argv[2]  # output model path

    loss = 'binary_crossentropy'
    metrics = ['binary_crossentropy']
    optimizer = Adam(0.00006)
    n_epoch = 10000
    batch_size = 128
    early_stop_patience = 10
    n_val = 10000

    try:
        eprint('try to load model from {}'.format(model_path))
        encoder = SNN.load(model_path)
    except Exception as e:
        eprint('load model error: {}'.format(e))
        eprint('attempt to load model failed, prepare to train it.')
        encoder = SNN(model_path=model_path)

    encoder.loss = loss
    encoder.metrics = metrics
    encoder.optimizer = optimizer
    encoder.n_epoch = n_epoch
    encoder.batch_size = batch_size
    encoder.early_step_patience = 10
    encoder.n_val = n_val

    encoder.fit(generator)
    encoder.save(model_path)
    eprint('model is trained and saved')

    backend.clear_session()
