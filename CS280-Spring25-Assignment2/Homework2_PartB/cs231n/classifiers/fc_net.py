from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *

class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. The architecture is:
    affine - ReLU - affine - softmax.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input.
        - hidden_dim: An integer giving the size of the hidden layer.
        - num_classes: An integer giving the number of classes to classify.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Initialize the weights and biases for the first layer.
        # Weights are drawn from a Gaussian distribution scaled by weight_scale,
        # and biases are initialized to zero.
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        
        # Initialize the weights and biases for the second layer.
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores.
        
        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss.
        - grads: Dictionary mapping parameter names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Unpack parameters
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        
        # Shape Alignment
        X_flat = X.reshape(X.shape[0], -1)
        hidden = np.dot(X_flat, W1) + b1

        # Forward pass: compute the class scores.
        # hidden = np.dot(X, W1) + b1          # affine transformation for layer 1
        hidden_relu = np.maximum(0, hidden)    # ReLU activation
        scores = np.dot(hidden_relu, W2) + b2  # affine transformation for layer 2

        # If y is None, then we are in test mode so just return scores.
        if y is None:
            return scores

        loss, grads = 0, {}
        # Compute the softmax loss.
        shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        N = X.shape[0]
        correct_logprobs = -np.log(probs[np.arange(N), y])
        data_loss = np.sum(correct_logprobs) / N
        reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2))
        loss = data_loss + reg_loss

        # Backward pass: compute gradients.
        dscores = probs
        dscores[np.arange(N), y] -= 1
        dscores /= N

        # Gradients for second layer parameters.
        grads['W2'] = np.dot(hidden_relu.T, dscores) + self.reg * W2
        grads['b2'] = np.sum(dscores, axis=0)

        # Backprop into the hidden layer.
        dhidden = np.dot(dscores, W2.T)
        # Backprop the ReLU non-linearity: zero gradient where hidden was less than or equal to 0.
        dhidden[hidden <= 0] = 0

        # Gradients for first layer parameters.
        X_flat = X.reshape(X.shape[0], -1)
        grads['W1'] = np.dot(X_flat.T, dhidden) + self.reg * W1
        grads['b1'] = np.sum(dhidden, axis=0)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture is:

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    Learnable parameters are stored in the self.params dictionary.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout.
        - normalization: Type of normalization to use: "batchnorm", "layernorm", or None.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random initialization.
        - dtype: Numpy datatype object; all computations will use this datatype.
        - seed: If not None, pass this random seed to the dropout layers.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)  # total number of layers (hidden + output)
        self.dtype = dtype
        self.params = {}

        ###########################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # Build a list of layer dimensions: input -> hidden layers -> output
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(1, len(layer_dims)):
            self.params['W' + str(i)] = weight_scale * np.random.randn(layer_dims[i-1], layer_dims[i])
            self.params['b' + str(i)] = np.zeros(layer_dims[i])
            if self.normalization is not None and i < len(layer_dims) - 1:
                # Initialize scale and shift parameters for normalization
                self.params['gamma' + str(i)] = np.ones(layer_dims[i])
                self.params['beta' + str(i)] = np.zeros(layer_dims[i])

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
      """
      Compute loss and gradient for the fully-connected net.

      Input / output: Same as TwoLayerNet above.
      """
      X = X.astype(self.dtype)
      mode = "test" if y is None else "train"

      # Set train/test mode for batchnorm params and dropout param since they
      # behave differently during training and testing.
      if self.use_dropout:
          self.dropout_param["mode"] = mode
      if self.normalization == "batchnorm":
          for bn_param in self.bn_params:
              bn_param["mode"] = mode
      scores = None
      ############################################################################
      # TODO: Implement the forward pass for the fully-connected net, computing  #
      # the class scores for X and storing them in the scores variable.          #
      #                                                                          #
      # When using dropout, you'll need to pass self.dropout_param to each       #
      # dropout forward pass.                                                    #
      #                                                                          #
      # When using batch normalization, you'll need to pass self.bn_params[0] to #
      # the forward pass for the first batch normalization layer, pass           #
      # self.bn_params[1] to the forward pass for the second batch normalization #
      # layer, etc.                                                              #
      ############################################################################
      # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
      caches = {}  # to store caches for each layer
      out = X

      # Forward pass for all hidden layers
      for i in range(1, self.num_layers):
          Wi = self.params['W' + str(i)]
          bi = self.params['b' + str(i)]
          # Affine forward
          out, fc_cache = affine_forward(out, Wi, bi)
          # Normalization (if applicable)
          if self.normalization == "batchnorm":
              gamma = self.params['gamma' + str(i)]
              beta = self.params['beta' + str(i)]
              bn_param = self.bn_params[i-1]
              out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
              fc_cache = (fc_cache, bn_cache)  # pack the affine and bn caches
          elif self.normalization == "layernorm":
              gamma = self.params['gamma' + str(i)]
              beta = self.params['beta' + str(i)]
              ln_param = self.bn_params[i-1]
              out, ln_cache = layernorm_forward(out, gamma, beta, ln_param)
              fc_cache = (fc_cache, ln_cache)
          # ReLU activation
          out, relu_cache = relu_forward(out)
          layer_cache = (fc_cache, relu_cache)
          # Dropout (if applicable)
          if self.use_dropout:
              out, dropout_cache = dropout_forward(out, self.dropout_param)
              layer_cache = (layer_cache, dropout_cache)
          caches[i] = layer_cache

      # Forward pass for the final layer (output layer, no ReLU/normalization/dropout)
      Wi = self.params['W' + str(self.num_layers)]
      bi = self.params['b' + str(self.num_layers)]
      scores, final_cache = affine_forward(out, Wi, bi)
      caches[self.num_layers] = final_cache

      # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
      ############################################################################
      #                             END OF YOUR CODE                             #
      ############################################################################

      # If test mode return early
      if mode == "test":
          return scores

      loss, grads = 0.0, {}
      ############################################################################
      # TODO: Implement the backward pass for the fully-connected net. Store the #
      # loss in the loss variable and gradients in the grads dictionary. Compute #
      # data loss using softmax, and make sure that grads[k] holds the gradients #
      # for self.params[k]. Don't forget to add L2 regularization!               #
      #                                                                          #
      # When using batch/layer normalization, you don't need to regularize the scale   #
      # and shift parameters.                                                    #
      #                                                                          #
      # NOTE: To ensure that your implementation matches ours and you pass the   #
      # automated tests, make sure that your L2 regularization includes a factor #
      # of 0.5 to simplify the expression for the gradient.                      #
      ############################################################################
      # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
      # Compute softmax loss and initial gradient
      loss, dscores = softmax_loss(scores, y)

      # Add L2 regularization for all layers
      for i in range(1, self.num_layers + 1):
          Wi = self.params['W' + str(i)]
          loss += 0.5 * self.reg * np.sum(Wi * Wi)

      # Backward pass for the final layer
      dout, dW, db = affine_backward(dscores, caches[self.num_layers])
      grads['W' + str(self.num_layers)] = dW + self.reg * self.params['W' + str(self.num_layers)]
      grads['b' + str(self.num_layers)] = db

      # Backward pass for hidden layers
      for i in range(self.num_layers - 1, 0, -1):
          cache_i = caches[i]
          # If dropout was used, unpack its cache and backprop through dropout first.
          if self.use_dropout:
              (layer_cache, dropout_cache) = cache_i
              dout = dropout_backward(dout, dropout_cache)
              fc_relu_cache = layer_cache
          else:
              fc_relu_cache = cache_i

          # Unpack caches for affine and ReLU (and possibly normalization)
          if self.normalization in ["batchnorm", "layernorm"]:
              (fc_cache, norm_cache), relu_cache = fc_relu_cache
              da = relu_backward(dout, relu_cache)
              # Backprop through normalization
              if self.normalization == "batchnorm":
                  da, dgamma, dbeta = batchnorm_backward(da, norm_cache)
              else:
                  da, dgamma, dbeta = layernorm_backward(da, norm_cache)
              dout, dW, db = affine_backward(da, fc_cache)
              grads['gamma' + str(i)] = dgamma
              grads['beta' + str(i)] = dbeta
          else:
              fc_cache, relu_cache = fc_relu_cache
              da = relu_backward(dout, relu_cache)
              dout, dW, db = affine_backward(da, fc_cache)

          grads['W' + str(i)] = dW + self.reg * self.params['W' + str(i)]
          grads['b' + str(i)] = db

      # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
      ############################################################################
      #                             END OF YOUR CODE                             #
      ############################################################################

      return loss, grads