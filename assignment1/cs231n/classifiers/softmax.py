import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros(W.shape)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
    scores = X[i].dot(W)  # (1,C)

    scores -= np.max(scores) # subtract max score to avoid instability

    #correct_class_prob = np.exp(scores[y[i]])
    denom = 0
    for j in xrange(num_classes):
      denom += np.exp(scores[j])
    #loss += -np.log(correct_class_prob / denom)
    loss += -scores[y[i]] + np.log(denom)

    for j in xrange(num_classes):
      prob = np.exp(scores[j])/denom
      if (j == y[i]):
        prob -= 1
      dW[:,j] += prob * X[i]

  loss /= num_train
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  scores = np.dot(X,W) # (N,C)
  scores -= np.amax(scores,axis=1, keepdims=True)
  exp_scores = np.exp(scores)
  probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
  corect_logprobs = -np.log(probs[range(num_train),y])

  data_loss = np.sum(corect_logprobs)/num_train
  reg_loss = 0.5*reg*np.sum(W*W)
  loss = data_loss + reg_loss

  dscores = probs
  dscores[range(num_train),y] -= 1
  dscores /= num_train

  dW = np.dot(X.T, dscores)
  dW += reg*W # don't forget the regularization gradient

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
  