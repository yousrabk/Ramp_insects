# All the imports
import os
os.environ["THEANO_FLAGS"] = "device=gpu"

from sklearn.pipeline import make_pipeline
from caffezoo.googlenet import GoogleNet
from nolearn.lasagne import NeuralNet, BatchIterator
from nolearn.lasagne.handlers import EarlyStopping
from lasagne import layers, nonlinearities
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from skimage.transform import rotate
from sklearn.cross_validation import KFold

class ClassifierFlipRotate(BaseEstimator):
    def __init__(self):
        self.net = build_model(hyper_parameters)
 
    def preprocess(self, X):
        X = (X / 255.)
        X = X.astype(np.float32)
        X = X.transpose((0, 3, 1, 2))
        return X
 
    def preprocess_y(self, y):
        return y.astype(np.int32)
 
    def fit(self, X, y):
        X = self.preprocess(X)
        self.net.fit(X, self.preprocess_y(y))
        return self
 
    def predict(self, X):
        X = self.preprocess(X)
        return self.net.predict(X)

    def predict_proba(self, X):
        X = self.preprocess(X)
        return self.net.predict_proba(X)


class FlipRotateBatchIterator(BatchIterator):
    def transform(self, Xb, yb):
        Xb, yb = super(FlipRotateBatchIterator, self).transform(Xb, yb)
        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        # indices = np.random.choice(bs, bs / 2, replace=False)
        cv = KFold(bs, 3, shuffle=True)
        for ii, (_, indices) in enumerate(cv):
            if ii == 0:
                Xb[indices] = Xb[indices, :, ::-1, :]
            if ii == 1:
                for index in indices:
                    Xb[index] = rotate(Xb[index, ...], 90)
            if ii == 2:
                for index in indices:
                    Xb[index] = rotate(Xb[index, ...], 180)

        return Xb, yb

hyper_parameters = dict(
    conv1_num_filters=64, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=128, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(4, 4),
    hidden4_num_units=500, hidden5_num_units=500,
    dropout5_p=0.5,
    output_num_units=18, output_nonlinearity=nonlinearities.softmax,
    update_learning_rate=0.01,
    update_momentum=0.9,
    max_epochs=20,
    on_epoch_finished=[EarlyStopping(patience=20, criterion='valid_accuracy', criterion_smaller_is_better=False)],
    batch_iterator_train=FlipRotateBatchIterator(batch_size=256 * 3)
)


def build_model(hyper_parameters):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('dropout5', layers.DropoutLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 3, 64, 64),
        use_label_encoder=True,
        verbose=1,
        **hyper_parameters
        )
    return net
