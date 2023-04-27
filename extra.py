import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x1,x1x2,x2,x0,e,M=1,0.5,5,(0,0.5),(0.15,0.2),10
#x1,x1x2,x2,x0,e,M=2,1,1,(0.5,1),(0.1,0.15),10
sns.set_theme(style="whitegrid")

x, y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
z = x1 * (x ** 2) + x1x2 *x*y + (y**2) * x2 -9
ax.contour(x, y, (x1 * (x ** 2) + x1x2 *x*y + (y**2) * x2 -9))

plt.show()