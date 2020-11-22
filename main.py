from model import *

if __name__ == '__main__':
    cnn = CNNModel()
    cnn.predict()
    lstm = LSTModel()
    lstm.predict()
    mlp = MLPModel()
    mlp.predict()

