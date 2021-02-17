import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        # raise Exception("Not implemented!")
        self.layers = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output)
        ]
        # end

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        # raise Exception("Not implemented!")
        W1 = self.layers[0].params()['W']
        W2 = self.layers[2].params()['W']

        W1.grad = np.zeros_like(W1.grad)
        W2.grad = np.zeros_like(W2.grad)
        # end

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        # raise Exception("Not implemented!")
        out_1 = self.layers[0].forward(X)
        out_2 = self.layers[1].forward(out_1)
        z_scores = self.layers[2].forward(out_2)

        reg_loss_1, dW1 = l2_regularization(W1.value, self.reg)
        reg_loss_2, dW2 = l2_regularization(W2.value, self.reg)

        ce_loss, dL_dZ = softmax_with_cross_entropy(z_scores, y)

        dL_dO2 = self.layers[2].backward(dL_dZ)
        dL_dO1 = self.layers[1].backward(dL_dO2)
        dL_dX = self.layers[0].backward(dL_dO1)
        W1.grad += dW1
        W2.grad += dW2

        loss = ce_loss + reg_loss_1 + reg_loss_2
        # end

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        # raise Exception("Not implemented!")
        out_1 = self.layers[0].forward(X)
        out_2 = self.layers[1].forward(out_1)
        z_scores = self.layers[2].forward(out_2)

        probs = softmax(z_scores)
        pred = np.argmax(probs, axis=1)
        # end
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        # raise Exception("Not implemented!")
        result = {
            'W1': self.layers[0].params()['W'],
            'B1': self.layers[0].params()['B'],
            'W2': self.layers[2].params()['W'],
            'B2': self.layers[2].params()['B']
        }
        # end

        return result
