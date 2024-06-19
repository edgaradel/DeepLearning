import matplotlib.pyplot as plt

X = [0.997, 2.643, 0.354, 0.075, 1.0, 0.03, 2.39, 0.364, 0.221, 0.437]
Y = [15.487507, 2.320735, 0.085742, 0.303032, 1.0, 0.025435, 4.436435,
     0.025435, 0.000503, 2.320735]
plt.figure()
plt.scatter(X,Y)
plt.xscale('log')
plt.yscale('log')
plt.title('scatter - scale last')
plt.show()
