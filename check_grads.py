import numpy as np
import layers

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def eval_numerical_gradient(f, x, df, h=1e-5):
  """
  Evaluate a numeric gradient for a function that accepts a numpy
  array and returns a numpy array.
  """
  grad = np.zeros_like(x)
  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    ix = it.multi_index
    
    oldval = x[ix]
    x[ix] = oldval + h
    pos = f(x).copy()
    x[ix] = oldval - h
    neg = f(x).copy()
    x[ix] = oldval
    
    grad[ix] = np.sum((pos - neg) * df) / (2 * h)
    it.iternext()
  return grad

def grad_check(layer, x, grad_out):
  grad_x_num = eval_numerical_gradient(layer.forward, x, grad_out)
  grad_x = layer.backward_inputs(grad_out)
  print("Relative error = ", rel_error(grad_x_num, grad_x))
  print("Error norm = ", np.linalg.norm(grad_x_num - grad_x))

print("Convolution")
x = np.random.randn(4, 3, 5, 5)
grad_out = np.random.randn(4, 2, 5, 5)
conv = layers.Convolution(x, 2, 3, "conv1")
grad_check(conv, x, grad_out)

print("\nMaxPooling")
x = np.random.randn(5, 4, 8, 8)
grad_out = np.random.randn(5, 4, 4, 4)
pool = layers.MaxPooling(x, "pool", 2, 2)
grad_check(pool, x, grad_out)

print("\nReLU")
x = np.random.randn(4, 3, 5, 5)
grad_out = np.random.randn(4, 3, 5, 5)
relu = layers.ReLU(x, "relu")
grad_check(relu, x, grad_out)

print("\nFC")
x = np.random.randn(20, 40)
grad_out = np.random.randn(20, 30)
fc = layers.FC(x, 30, "fc")
grad_check(fc, x, grad_out)

print("\nSoftmaxCrossEntropyWithLogits")
x = np.random.randn(50, 20)
y = np.zeros([50, 20])
y[:,0] = 1
loss = layers.SoftmaxCrossEntropyWithLogits()
grad_x_num = eval_numerical_gradient(lambda x: loss.forward(x, y), x, 1)
out = loss.forward(x, y)
grad_x = loss.backward_inputs()
print("Relative error = ", rel_error(grad_x_num, grad_x))
print("Error norm = ", np.linalg.norm(grad_x_num - grad_x))

