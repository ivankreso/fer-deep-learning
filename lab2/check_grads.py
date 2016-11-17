import numpy as np
import layers

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def eval_numerical_gradient(f, x, df, h=1e-5):
  """
  Evaluate a numeric gradient for a function that accepts a numpy
  array and returns a numpy array.
  - f should be a function that takes a single argument
  - x is the point (numpy array) to evaluate the gradient at
  """
  grad = np.zeros_like(x)
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    ix = it.multi_index
    
    oldval = x[ix]
    x[ix] = oldval + h
    # evaluate f(x + h)
    pos = f(x.copy()).copy()
    x[ix] = oldval - h
    # evaluate f(x - h)
    neg = f(x.copy()).copy()
    x[ix] = oldval
    
    # compute the partial derivative with centered formula
    grad[ix] = np.sum((pos - neg) * df) / (2 * h)
    # step to next dimension
    it.iternext()
  return grad

def check_grad_inputs(layer, x, grad_out):
  """
  Args:
    layer: Layer object
    x: ndarray tensor input data
    grad_out: ndarray tensor gradient from the next layer
  """
  grad_x_num = eval_numerical_gradient(layer.forward, x, grad_out)
  grad_x = layer.backward_inputs(grad_out)
  print("Relative error = ", rel_error(grad_x_num, grad_x))
  print("Error norm = ", np.linalg.norm(grad_x_num - grad_x))

def check_grad_params(layer, x, w, b, grad_out):
  """
  Args:
    layer: Layer object
    x: ndarray tensor input data
    w: ndarray tensor layer weights
    b: ndarray tensor layer biases
    grad_out: ndarray tensor gradient from the next layer
  """
  func = lambda params: layer.forward(x)
  grad_w_num = eval_numerical_gradient(func, w, grad_out)
  grad_b_num = eval_numerical_gradient(func, b, grad_out)
  grads = layer.backward_params(grad_out)
  grad_w = grads[0][1]
  grad_b = grads[1][1]
  print("Check weights:")
  print("Relative error = ", rel_error(grad_w_num, grad_w))
  print("Error norm = ", np.linalg.norm(grad_w_num - grad_w))
  print("Check biases:")
  print("Relative error = ", rel_error(grad_b_num, grad_b))
  print("Error norm = ", np.linalg.norm(grad_b_num - grad_b))

print("Convolution")
x = np.random.randn(4, 3, 5, 5)
grad_out = np.random.randn(4, 2, 5, 5)
conv = layers.Convolution(x, 2, 3, "conv1")
print("Check grad wrt input")
check_grad_inputs(conv, x, grad_out)
print("Check grad wrt params")
check_grad_params(conv, x, conv.weights, conv.bias, grad_out)

print("\nMaxPooling")
x = np.random.randn(5, 4, 8, 8)
grad_out = np.random.randn(5, 4, 4, 4)
pool = layers.MaxPooling(x, "pool", 2, 2)
print("Check grad wrt input")
check_grad_inputs(pool, x, grad_out)

print("\nReLU")
x = np.random.randn(4, 3, 5, 5)
grad_out = np.random.randn(4, 3, 5, 5)
relu = layers.ReLU(x, "relu")
print("Check grad wrt input")
check_grad_inputs(relu, x, grad_out)

print("\nFC")
x = np.random.randn(20, 40)
grad_out = np.random.randn(20, 30)
fc = layers.FC(x, 30, "fc")
print("Check grad wrt input")
check_grad_inputs(fc, x, grad_out)
print("Check grad wrt params")
check_grad_params(fc, x, fc.weights, fc.bias, grad_out)

print("\nSoftmaxCrossEntropyWithLogits")
x = np.random.randn(50, 20)
y = np.zeros([50, 20])
y[:,0] = 1
loss = layers.SoftmaxCrossEntropyWithLogits()
grad_x_num = eval_numerical_gradient(lambda x: loss.forward(x, y), x, 1)
out = loss.forward(x, y)
grad_x = loss.backward_inputs(x, y)
print("Relative error = ", rel_error(grad_x_num, grad_x))
print("Error norm = ", np.linalg.norm(grad_x_num - grad_x))

print("\nL2Regularizer")
x = np.random.randn(5, 4, 8, 8)
grad_out = np.random.randn(5, 4, 4, 4)
l2reg = layers.L2Regularizer(x, 1e-2, 'L2reg')
print("Check grad wrt params")
func = lambda params: l2reg.forward()
grad_num = eval_numerical_gradient(func, l2reg.weights, 1)
grads = l2reg.backward_params()
grad = grads[0][1]
print("Relative error = ", rel_error(grad_num, grad))
print("Error norm = ", np.linalg.norm(grad_num - grad))
