# Simple gradient descent algorithm

# linear equation: y = wx+b
# loss = (y-yhat)**2 / N

import numpy as np

x = np.random.randn(10,1)

w0 = np.float32(np.random.rand())
b0 = np.float32(np.random.rand())

y = w0*x + b0

w = 0.0
b = 0.0

learning_rate = 0.01
N = x.shape[0]
def descend(x,y,w,b,learning_rate):

    # L = (y - (wx+b))**2
    dldw = 0.0
    dldb = 0.0
    N = x.shape[0]

    for xi, yi in zip(x,y):
        xi = xi.item()
        yi = yi.item()
        dldw += -2*xi*(yi-(w*xi+b))
        dldb += -2*(yi-(w*xi+b))

    w = w - learning_rate*dldw*(1/N)
    b = b - learning_rate * dldb * (1/N)

    return w,b

# update

for epoch in range(400):

    w,b = descend(x,y,w,b,learning_rate)
    yhat = w*x+b
    loss = np.divide(np.sum((y - yhat)**2, axis=0), x.shape[0])
    # print(f"At epoch {epoch}, loss is {loss}. w: {w}, b: {b}")


true_w = w0
true_b = b0

learned_w = w
learned_b = b

print(f"True model      : y = {true_w:.4f} * x + {true_b:.4f}")
print(f"Learned model   : y = {learned_w:.4f} * x + {learned_b:.4f}")