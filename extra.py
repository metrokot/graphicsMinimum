import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
x1,x1x2,x2,x0,e,M=1,0.5,5,(0,0.5),(0.15,0.2),10
func = lambda x,y: x1 * (x ** 2) + x1x2 *x*y + (y**2) * x2
C = [4,3,2,1]
x = np.linspace(-2, 2, 50)
y = np.linspace(-2, 2, 50)
X, Y = np.meshgrid(x, y)
plt.contour(X, Y, func(X, Y), levels=[i for i in C])
plt.show()

