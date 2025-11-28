import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

y_0 = 1.8
Data = np.array([0.54, 0.48, 0.53, 0.48, 0.53]) # T values
EndHeight = np.zeros(len(Data))

def StandardDev(array):
    mean = np.mean(Data)
    a = 0
    for v in range(len(array)):
        a += (array[v] - mean)**2
    return np.sqrt(a/(len(array)-1)), mean

sd, mean = StandardDev(Data)
print(f'{mean} +/- {sd}')

# Y0 = 1.8
# Y_data = 0

def Height_Time(t, g):
    return y_0 + (g*t**2)/2

# We expect height_time(t, g) = 0

popt, pcov = curve_fit(Height_Time, Data, EndHeight)
print(f'm/s^2 = {np.round(popt, 1)} +/- {np.round(np.sqrt(pcov[0]), 1)}')

y_0 = 1.8
Data = np.array([0.54, 0.48, 0.53, 0.48, 0.53, 0.6, 0.6, 0.6, 0.54, 0.67, 0.6, 0.55, 0.54, 0.54, 0.6, 0.53]) # T values
EndHeight = np.zeros(len(Data))

def StandardDev(array):
    mean = np.mean(Data)
    a = 0
    for v in range(len(array)):
        a += (array[v] - mean)**2
    return np.sqrt(a/(len(array)-1)), mean

sd, mean = StandardDev(Data)
print(f'{mean} +/- {sd}')

# Y0 = 1.8
# Y_data = 0

def Height_Time(t, g):
    return y_0 + (g*t**2)/2

# We expect height_time(t, g) = 0

popt, pcov = curve_fit(Height_Time, Data, EndHeight)
print(f'm/s^2 = {np.round(popt, 1)} +/- {np.round(np.sqrt(pcov[0]), 1)}')

plt.hist(Data, bins=5)
plt.show()