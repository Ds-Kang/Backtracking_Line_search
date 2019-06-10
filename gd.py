import numpy as np

def min_gd(fun, x0, grad, args=()):
	while np.linalg.norm(grad(x0,*args))>0.0001:
		alpha=0.3
		beta=0.8

		d_x=-grad(x0,*args)
		t=1
		while fun(x0+t*d_x,*args) > fun(x0,*args)+alpha*t*np.dot(grad(x0,*args).T,d_x):
			t=beta*t

		x0+=t*d_x
	return x0