from __future__ import division, absolute_import
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.estimator import regression
from os.path import isfile, join


class EMR:
    def __init__(self):
        self.target_classes = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

    def build_network(self):
        # Smaller 'AlexNet'
        # https://github.com/tflearn/tflearn/blob/master/examples/images/alexnet.py
        # tutorial link : https://github.com/tensorflow/models/tree/master/tutorials/image/alexnet
        print("____Building Neural Network_____")
        self.network = input_data(shape=[None, 48, 48, 1])
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        # Relu kya hota hai ?
        # bacha loge : https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
        self.network = max_pool_2d(self.network, 3, strides=2)
        # self.network = local_response_normalization(self.network)
        self.network = conv_2d(self.network, 64, 5, activation='relu')
        self.network = max_pool_2d(self.network, 3, strides=2)
        self.network = conv_2d(self.network, 128, 4, activation='relu')
        self.network = dropout(self.network, 0.3)
        self.network = fully_connected(self.network, 3072, activation='relu')
        self.network = fully_connected(self.network, len(self.target_classes), activation='softmax')
        self.network = regression(self.network,
                                  optimizer='momentum',
                                  loss='categorical_crossentropy')
        self.model = tflearn.DNN(
            self.network,
            checkpoint_path='model_1_nimish',
            max_checkpoints=1,
            tensorboard_verbose=2
        )
        self.load_model()

    def predict(self, image):
        if image is None:
            return None
        image = image.reshape([-1, 48, 48, 1])
        return self.model.predict(image)

    def load_model(self):
        if isfile("C:/Users/ashis/PycharmProjects/Py/emote2emoji/source/model_1_nimish.tflearn.meta"):
            self.model.load("C:/Users/ashis/PycharmProjects/Py/emote2emoji/source/model_1_nimish.tflearn")
            print('Loading pre-trained offline model')
        else:
            print("Please bhai kaam kar le")


if __name__ == "__main__":
    network = EMR()
    import opencv3
