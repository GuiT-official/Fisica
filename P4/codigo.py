import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot(title, num, R, Rvar = None, C = None):
    data = pd.read_csv(rf'E:/osu{num}.csv', skiprows=4)

    plt.title(title)
    plt.xlabel('Log_10(Frequência)')
    plt.ylabel('dB/graus')

    X = np.linspace(min(data['Freq(Hz)']), max(data['Freq(Hz)']), int(1e6))
    if Rvar == None:
        g = lambda x: 20*np.log10(1/np.sqrt(1 + (2*np.pi*x*R*C)**2))
        f = lambda x: np.rad2deg(np.arctan(-2*np.pi*x*R*C))
    else:
        g = lambda x: [20*np.log10(Rvar/R)]*len(x)
        f = lambda x: [180]*len(x)

    plt.plot(np.log10(data['Freq(Hz)']), data['Gain(dB)'], color='blue')
    plt.plot(np.log10(X), g(X), color='green')

    plt.plot(np.log10(data['Freq(Hz)']), data['Phase'], color='red')
    plt.plot(np.log10(X), f(X), color='orange')
    plt.show()

if __name__ == '__main__':
    plot('Bode - Max Amplitude', 0, R=997, Rvar=10410)
    plot('Bode - 1/2 da Max Amplitude', 1, R=997, Rvar=5220)
    plot('Bode - parte 2', 2, R=984, C=97e-9)
