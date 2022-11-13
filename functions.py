import numpy as np

# 𝑑𝐱=𝜎^𝑡𝑑𝐰 인 경우, 
def weight_function(sigma, t):
    return 1/(2* np.log(sigma)) * (sigma**(2*t) - 1)