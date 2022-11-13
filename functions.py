import numpy as np

# 𝑑𝐱=𝜎^𝑡𝑑𝐰 인 경우, 
# https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing#scrollTo=YyQtV7155Nht
# the weighting function  𝜆(𝑡)=(𝜎^{2𝑡}−1)/(2log𝜎)
def weight_function(t, sigma):
    return (pow(sigma, 2*t) - 1) / (2 * np.log(sigma))


def unit_test():
    import matplotlib.pylab as plt
    sigma = 0.5
    x = list(range(10))
    y = [ weight_function(i, sigma) for i in x]
    plt.plot(x, y); plt.show()

