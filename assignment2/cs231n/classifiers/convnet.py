import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class FourLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  conv - relu - 2x2 max pool X 2 - affine - relu - affine - softmax
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=16, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C = input_dim[0]
    H = input_dim[1]
    W = input_dim[2]
    F = num_filters
    input_size = input_dim[0]*input_dim[1]*input_dim[2]
    self.params['W1'] = weight_scale * np.random.randn(F,C,filter_size,filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale * np.random.randn(F,F,filter_size,filter_size)
    self.params['b2'] = np.zeros(num_filters)
    self.params['W3'] = weight_scale * np.random.randn(F*H*W/16, hidden_dim)
    self.params['b3'] = np.zeros(hidden_dim)
    self.params['W4'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b4'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv1, ccache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    #print conv1.shape
    conv2, ccache2 = conv_relu_pool_forward(conv1, W2, b2, conv_param, pool_param)
    #print conv2.shape
    cshape = conv2.shape
    hidden, hcache = affine_relu_forward(conv2.reshape((X.shape[0],-1)), W3, b3)
    scores, scache = affine_forward(hidden, W4, b4)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores,y)
    reg = self.reg
    reg_loss = 0.5*reg*(np.sum(W4*W4)+np.sum(W3*W3)+np.sum(W2*W2)+np.sum(W1*W1))
    loss = data_loss+reg_loss

    dhidden, dW4, db4 = affine_backward(dscores, scache)
    dconv2, dW3, db3 = affine_relu_backward(dhidden, hcache)
    #print dconv2.shape
    dconv1, dW2, db2 = conv_relu_pool_backward(dconv2.reshape((cshape)), ccache2)
    #print dconv1.shape
    dx, dW1, db1 = conv_relu_pool_backward(dconv1, ccache1)

    dW4 += reg * W4
    dW3 += reg * W3
    dW2 += reg * W2
    dW1 += reg * W1

    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['W3'] = dW3
    grads['W4'] = dW4
    grads['b1'] = db1.reshape((-1))
    grads['b2'] = db2.reshape((-1))
    grads['b3'] = db3.reshape((-1))
    grads['b4'] = db4.reshape((-1))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
class FiveLayerConvNet(object):
  """
  A five-layer convolutional network with the following architecture:

  (conv - relu)x2 conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C = input_dim[0]
    H = input_dim[1]
    W = input_dim[2]
    F = num_filters
    input_size = input_dim[0]*input_dim[1]*input_dim[2]
    ws = np.sqrt(2./(C*H*W+F*H*W))
    self.params['W0'] = ws * np.random.randn(F,C,filter_size,filter_size)
    self.params['b0'] = np.zeros(num_filters)
    ws = np.sqrt(1./(F*H*W))
    self.params['W1'] = ws * np.random.randn(F,F,filter_size,filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = ws * np.random.randn(F,F,filter_size,filter_size)
    self.params['b2'] = np.zeros(num_filters)
    ws = np.sqrt(2./(F*H*W/4 + hidden_dim))
    self.params['W3'] = ws * np.random.randn(F*H*W/4, hidden_dim)
    self.params['b3'] = np.zeros(hidden_dim)
    ws = np.sqrt(2./(num_classes + hidden_dim))
    self.params['W4'] = ws * np.random.randn(hidden_dim, num_classes)
    self.params['b4'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W0, b0 = self.params['W0'], self.params['b0']
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    W4, b4 = self.params['W4'], self.params['b4']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv0, ccache0 = conv_relu_forward(X, W0, b0, conv_param)
    conv1, ccache1 = conv_relu_forward(conv0, W1, b1, conv_param)
    #print conv1.shape
    conv2, ccache2 = conv_relu_pool_forward(conv1, W2, b2, conv_param, pool_param)
    #print conv2.shape
    cshape = conv2.shape
    hidden, hcache = affine_relu_forward(conv2.reshape((X.shape[0],-1)), W3, b3)
    scores, scache = affine_forward(hidden, W4, b4)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores,y)
    reg = self.reg
    reg_loss = 0.5*reg*(np.sum(W4*W4)+np.sum(W3*W3)+np.sum(W2*W2)+np.sum(W1*W1))
    loss = data_loss+reg_loss

    dhidden, dW4, db4 = affine_backward(dscores, scache)
    dconv2, dW3, db3 = affine_relu_backward(dhidden, hcache)
    #print dconv2.shape
    dconv1, dW2, db2 = conv_relu_pool_backward(dconv2.reshape((cshape)), ccache2)
    #print dconv1.shape
    dconv0, dW1, db1 = conv_relu_backward(dconv1, ccache1)
    dx, dW0, db0 = conv_relu_backward(dconv0, ccache0)

    dW4 += reg * W4
    dW3 += reg * W3
    dW2 += reg * W2
    dW1 += reg * W1
    dW0 += reg * W0

    grads['W0'] = dW0
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['W3'] = dW3
    grads['W4'] = dW4
    grads['b0'] = db0.reshape((-1))
    grads['b1'] = db1.reshape((-1))
    grads['b2'] = db2.reshape((-1))
    grads['b3'] = db3.reshape((-1))
    grads['b4'] = db4.reshape((-1))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
  """
  Convenience layer that perorms an affine transform followed by BN, ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(bn)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache

def conv_bn_relu_forward(x, w, b, conv_param, gamma, beta, bn_param):
  """
  Convenience layer that perorms an conv transform followed by BN, ReLU
  
  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer 

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = conv_forward_fast(x, w, b, conv_param)
  bn, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  out, relu_cache = relu_forward(bn)
  cache = (fc_cache, bn_cache, relu_cache)
  return out, cache

def conv_bn_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
  """
  Convenience layer that perorms an conv transform followed by BN, ReLU
  
  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer 
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = conv_forward_fast(x, w, b, conv_param)
  bn, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
  r, relu_cache = relu_forward(bn)
  out, p_cache = max_pool_forward_fast(r, pool_param)
  cache = (fc_cache, bn_cache, relu_cache, p_cache)
  return out, cache

def affine_bn_relu_backward(dout, cache):
  """
  Backward pass for the conv-bn-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dbn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
  dx, dw, db = affine_backward(dbn, fc_cache)
  return dx, dw, db, dgamma, dbeta

def conv_bn_relu_backward(dout, cache):
  """
  Backward pass for the conv-bn-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dbn, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
  dx, dw, db = conv_backward_fast(dbn, fc_cache)
  return dx, dw, db, dgamma, dbeta

def conv_bn_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-bn-relu convenience layer
  """
  fc_cache, bn_cache, relu_cache, p_cache = cache
  ds = max_pool_backward_fast(dout, p_cache)
  da = relu_backward(ds, relu_cache)
  dbn, dgamma, dbeta = spatial_batchnorm_backward(da, bn_cache)
  dx, dw, db = conv_backward_fast(dbn, fc_cache)
  return dx, dw, db, dgamma, dbeta

class SevenLayerConvNet(object):
  """
  A Seven-layer convolutional network with the following architecture:

  (conv - relu - conv - relu - pool)x2 - (affine - relu)x2 - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
               hidden_dim2=500, hidden_dim1=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               use_batchnorm=1,N=50,dropout=0,seed=None,dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg 
    self.use_dropout = dropout > 0
    self.dtype = dtype
    self.num_layers = 7
    self.use_batchnorm = use_batchnorm
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C = input_dim[0]
    H = input_dim[1]
    W = input_dim[2]
    F = num_filters
    input_size = input_dim[0]*input_dim[1]*input_dim[2]
    ws = np.sqrt(2./(C*H*W+F*H*W))
    self.params['W0'] = ws * np.random.randn(F,C,filter_size,filter_size)
    self.params['b0'] = np.zeros(num_filters)
    ws = np.sqrt(1./(F*H*W))
    self.params['W1'] = ws * np.random.randn(F,F,filter_size,filter_size)
    self.params['b1'] = np.zeros(num_filters)
    ws = np.sqrt(1./(F*H*W/4.0))
    self.params['W2'] = ws * np.random.randn(F,F,filter_size,filter_size)
    self.params['b2'] = np.zeros(num_filters)
    self.params['W3'] = ws * np.random.randn(F,F,filter_size,filter_size)
    self.params['b3'] = np.zeros(num_filters)
    ws = np.sqrt(2./(F*H*W/16.0 + hidden_dim1))
    self.params['W4'] = ws * np.random.randn(F*H*W/16, hidden_dim1)
    self.params['b4'] = np.zeros(hidden_dim1)
    ws = np.sqrt(2./(hidden_dim2 + hidden_dim1))
    self.params['W5'] = ws * np.random.randn(hidden_dim1, hidden_dim2)
    self.params['b5'] = np.zeros(hidden_dim2)
    ws = np.sqrt(2./(num_classes + hidden_dim2))
    self.params['W6'] = ws * np.random.randn(hidden_dim2, num_classes)
    self.params['b6'] = np.zeros(num_classes)
    dims = []
    dims.append(F)
    dims.append(F)
    dims.append(F)
    dims.append(F)
    dims.append(hidden_dim1)
    dims.append(hidden_dim2)
    dims.append(num_classes)
    for i in xrange(self.num_layers):
        if self.use_batchnorm:
            if i != self.num_layers-1: # no bn in last layer
              self.params['gamma'+str(i)] = np.ones(dims[i])
              self.params['beta'+str(i)] = np.zeros(dims[i])

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train',
                         'running_mean': np.zeros(dims[i]),
                         'running_var': np.zeros(dims[i])}
                         for i in xrange(self.num_layers - 1)]


    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    mode = 'test' if y is None else 'train'

    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param['mode'] = mode


    W0, b0, gamma0, beta0 = self.params['W0'], self.params['b0'], self.params['gamma0'], self.params['beta0']
    W1, b1, gamma1, beta1 = self.params['W1'], self.params['b1'], self.params['gamma1'], self.params['beta1']
    W2, b2, gamma2, beta2 = self.params['W2'], self.params['b2'], self.params['gamma2'], self.params['beta2']
    W3, b3, gamma3, beta3 = self.params['W3'], self.params['b3'], self.params['gamma3'], self.params['beta3']
    W4, b4, gamma4, beta4 = self.params['W4'], self.params['b4'], self.params['gamma4'], self.params['beta4']
    W5, b5, gamma5, beta5 = self.params['W5'], self.params['b5'], self.params['gamma5'], self.params['beta5']
    W6, b6 = self.params['W6'], self.params['b6']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv0, ccache0 = conv_bn_relu_forward(X, W0, b0, conv_param,
                         gamma0,beta0,self.bn_params[0])
    conv1, ccache1 = conv_bn_relu_pool_forward(conv0, W1, b1, conv_param, pool_param,
                         gamma1,beta1,self.bn_params[1])
    #print conv1.shape
    conv2, ccache2 = conv_bn_relu_forward(conv1, W2, b2, conv_param,
                         gamma2,beta2,self.bn_params[2])
    conv3, ccache3 = conv_bn_relu_pool_forward(conv2, W3, b3, conv_param, pool_param,
                         gamma3,beta3,self.bn_params[3])
    #print conv2.shape
    cshape = conv3.shape
    hidden1, hcache1 = affine_bn_relu_forward(conv3.reshape((X.shape[0],-1)), W4, b4,
                         gamma4,beta4,self.bn_params[4])
    if self.use_dropout:
        hidden1, dcache1 = dropout_forward(hidden1, self.dropout_param)
    hidden2, hcache2 = affine_bn_relu_forward(hidden1, W5, b5,
                         gamma5,beta5,self.bn_params[5])
    if self.use_dropout:
        hidden2, dcache2 = dropout_forward(hidden2, self.dropout_param)
    scores, scache = affine_forward(hidden2, W6, b6)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    data_loss, dscores = softmax_loss(scores,y)
    reg = self.reg
    reg_loss = 0.5*reg*(np.sum(W6*W6)+np.sum(W5*W5)+np.sum(W4*W4) \
        +np.sum(W3*W3)+np.sum(W2*W2)+np.sum(W1*W1))
    loss = data_loss+reg_loss

    dhidden2, dW6, db6 = affine_backward(dscores, scache)
    if self.use_dropout:
        dhidden2 = dropout_backward(dhidden2, dcache2)
    dhidden1, dW5, db5, dgamma5, dbeta5 = affine_bn_relu_backward(dhidden2, hcache2)
    if self.use_dropout:
        dhidden1 = dropout_backward(dhidden1, dcache1)
    dconv3, dW4, db4, dgamma4, dbeta4  = affine_bn_relu_backward(dhidden1, hcache1)
    #print dconv2.shape
    dconv2, dW3, db3, dgamma3, dbeta3  = conv_bn_relu_pool_backward(dconv3.reshape((cshape)), ccache3)
    #print dconv1.shape
    dconv1, dW2, db2, dgamma2, dbeta2  = conv_bn_relu_backward(dconv2, ccache2)
    dconv0, dW1, db1, dgamma1, dbeta1  = conv_bn_relu_pool_backward(dconv1, ccache1)
    dx, dW0, db0, dgamma0, dbeta0 = conv_bn_relu_backward(dconv0, ccache0)

    dW6 += reg * W6
    dW5 += reg * W5
    dW4 += reg * W4
    dW3 += reg * W3
    dW2 += reg * W2
    dW1 += reg * W1
    dW0 += reg * W0

    grads['gamma0'] = dgamma0
    grads['gamma1'] = dgamma1
    grads['gamma2'] = dgamma2
    grads['gamma3'] = dgamma3
    grads['gamma4'] = dgamma4
    grads['gamma5'] = dgamma5
    grads['beta0'] = dbeta0
    grads['beta1'] = dbeta1
    grads['beta2'] = dbeta2
    grads['beta3'] = dbeta3
    grads['beta4'] = dbeta4
    grads['beta5'] = dbeta5
    grads['W0'] = dW0
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['W3'] = dW3
    grads['W4'] = dW4
    grads['W5'] = dW5
    grads['W6'] = dW6
    grads['b0'] = db0.reshape((-1))
    grads['b1'] = db1.reshape((-1))
    grads['b2'] = db2.reshape((-1))
    grads['b3'] = db3.reshape((-1))
    grads['b4'] = db4.reshape((-1))
    grads['b5'] = db5.reshape((-1))
    grads['b6'] = db6.reshape((-1))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads
  
pass
