import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sc

class Fisica:
    def __init__(self, arq=None, data=None, freq=None):
        self.freq = freq

        if isinstance(data,pd.DataFrame):
            self.data = data
        else:
            data = pd.read_csv(arq)
            data['Time(s)'] = data['Time(s)'].apply(lambda x: float(x))
            data['Time(s)'] -= data['Time(s)'][0]
            data['CH1V'] = data['CH1V'].apply(lambda x: float(x))
            data['CH2V'] = data['CH2V'].apply(lambda x: float(x))
            self.data = data

    def lim_sup(self, lim):
        return Fisica(data=self.data[self.data['Time(s)'] <= lim], freq=self.freq)
    
    def lim_inf(self, lim):
        return Fisica(data=self.data[lim <= self.data['Time(s)']], freq=self.freq)
    
    def bruto(self):
        plt.plot(self.data['Time(s)'], self.data['CH1V'], color='blue')
        plt.plot(self.data['Time(s)'], self.data['CH2V'], color='red')

    def fit(self):
        func = lambda t,Voff,V0,freq,phi: Voff + V0*np.sin(2*np.pi*freq*t + phi)

        def chute(ch):
            m = max(self.data[ch])
            u = (m + min(self.data[ch]))/2
            phi = np.arcsin((self.data[ch][0]-u)/(m-u)) - 2*np.pi*self.freq*self.data['Time(s)'][0]
            f = self.freq

            return np.array([u, m-u, f, phi])

        self.coef1 = sc.curve_fit(lambda t,phi: func(t, *chute('CH1V')[:3], phi),self.data['Time(s)'], self.data['CH1V'], p0=chute('CH1V')[3])[0]
        self.coef2 = sc.curve_fit(lambda t,phi: func(t, *chute('CH2V')[:3], phi),self.data['Time(s)'], self.data['CH2V'], p0=chute('CH2V')[3])[0]

        self.coef1 = sc.curve_fit(func,self.data['Time(s)'], self.data['CH1V'], p0=np.concatenate((chute('CH1V')[:3],self.coef2)))[0]
        self.coef2 = sc.curve_fit(func,self.data['Time(s)'], self.data['CH2V'], p0=np.concatenate((chute('CH2V')[:3],self.coef2)))[0]

        plt.plot(self.data['Time(s)'], func(self.data['Time(s)'], *self.coef1), color='green')
        plt.plot(self.data['Time(s)'], func(self.data['Time(s)'], *self.coef2), color='brown')

    def VR(self):
        plt.plot(self.data['Time(s)'], self.data['CH1V'] - self.data['CH2V'], color='black')

    @property
    def V0R(self):
        s = self.data['CH1V'] - self.data['CH2V']
        return (max(s)-min(s))/2

    @property
    def ganho(self):
        return 20*np.log10(self.coef2[1]/self.coef1[1])
    
    @property
    def fase(self):
        return self.coef2[3] - self.coef1[3]

    @property
    def f(self):
        return self.coef2[2]
    
class Fisica2:
    def __init__(self, path, scale, ini):
        if scale == 'linear':
            self.scale = lambda y: y
        elif scale == 'log':
            self.scale = lambda y: np.log10(y)

        self.ganho = np.array([0]*9)
        self.ganhoR = np.array([0]*9)
        self.fase = np.array([0]*9)
        self.faseR = np.array([0]*9)
        self.fre = np.array([0]*9)
        for k in range(9):
            a = Fisica(rf'{path}{ini*2**k}h0.csv', freq=ini*2**k)

            self.info(f'Gráfico com frequência nominal de {ini*2**k} Hz', 'Tempo (s)', 'Tensão (V)')
            a.bruto()
            a.fit()
            a.VR()
            self.ganho[k] = a.ganho
            self.ganhoR[k] = self.ganho_R(a)
            self.fase[k] = a.fase
            self.faseR[k] = self.fase_R(a)
            self.fre[k] = a.f
            plt.show()

    def ganho_R(self, a):
        return 20*np.log10(a.V0R/a.coef1[1])
    
    def fase_R(self, a):
        return np.arcsin((a.coef1[1]*np.sin(a.coef1[3]) - a.coef2[1]*np.sin(a.coef2[3]))/a.V0R) - a.coef1[3]
    
    def info(self, title, xlabel, ylabel):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    def bode(self, tipo, Ganho, Fase):
        X=np.linspace(self.fre[0],self.fre[8],int(1e6))

        self.info(f'Diagrama de Bode - Ganho {tipo}', 'Log_10(Frequência/Hz)', 'Ganho (DB)')
        plt.scatter(self.scale(self.fre), self.ganho, color='blue')
        plt.plot(self.scale(X), 20*np.log10(Ganho(X)), color='green')
        plt.show()

        self.info(f'Diagrama de Bode - Fase {tipo}', 'Log_10(Frequência/Hz)', 'Fase (graus)')
        plt.scatter(self.scale(self.fre), self.fase, color='red')
        plt.plot(self.scale(X), np.rad2deg(np.arctan(Fase(X))), color='brown')
        plt.show()

    def bodeR(self, Ganho, Fase):
        X=np.linspace(self.fre[0],self.fre[8],int(1e6))

        self.info(f'Diagrama de Bode - Ganho R', 'Log_10(Frequência/Hz)', 'Ganho (DB)')
        plt.scatter(self.scale(self.fre), self.ganhoR, color='blue')
        plt.plot(self.scale(X), 20*np.log10(Ganho(X)), color='green')
        plt.show()

        self.info(f'Diagrama de Bode - Fase R', 'Log_10(Frequência/Hz)', 'Fase (graus)')
        plt.scatter(self.scale(self.fre), self.faseR, color='red')
        plt.plot(self.scale(X), np.rad2deg(np.arctan(Fase(X))), color='brown')
        plt.show()

    def tabela(self):
        return pd.DataFrame({'Frequência': self.fre, 'Ganho': self.ganho, 'Fase': self.fase, 'GanhoR': self.ganhoR, 'FaseR': self.faseR})

class RC(Fisica2):
    def __init__(self, path=None, scale='linear', R=None, C=None):
        super().__init__(path, scale, 46)

        self.bode('C', lambda x: 1/np.sqrt(1+(2*np.pi*x*R*C)**2),
                       lambda x: -2*np.pi*x*R*C)
        self.bodeR(lambda x: 2*np.pi*x*R*C/np.sqrt(1+(2*np.pi*x*R*C)**2),
                   lambda x: 1/(2*np.pi*x*R*C))

class RL(Fisica2):
    def __init__(self, path=None, scale='linear', R=None, L=None):
        super().__init__(path, scale, 46)

        self.bode('L', lambda x: 2*np.pi*x*L/np.sqrt(R**2+(2*np.pi*x*L)**2),
                       lambda x: R/(2*np.pi*x*L))
        self.bodeR(lambda x: R/np.sqrt(R**2+(2*np.pi*x*L)**2),
                   lambda x: -(2*np.pi*x*L)/R)

class RLC(Fisica2):
    def __init__(self, path=None, scale='linear', R=None, L=None, C=None):
        super().__init__(path, scale, 220)

        self.bode('LC', lambda x: abs(2*np.pi*x*L-1/(2*np.pi*x*C))/np.sqrt(R**2+(2*np.pi*x*L-1/(2*np.pi*x*C))**2),
                        lambda x: R/(2*np.pi*x*L - 1/(2*np.pi*x*C)))
        self.bodeR(lambda x: R/np.sqrt(R**2+(2*np.pi*x*L-1/(2*np.pi*x*C))**2),
                   lambda x: -(2*np.pi*x*L - 1/(2*np.pi*x*C))/R)

#RC('C:/Users/Gabriel/Downloads/P3/RC/RigolDSrc', scale='log', R=2166, C=1e-7).tabela()
#RL('C:/Users/Gabriel/Downloads/P3/RL/1/RL1mh', scale='log', R=99+21, L=1e-3).tabela()
#RL('C:/Users/Gabriel/Downloads/P3/RL/5/RL5mh', scale='log', R=300+21*5, L=5e-3).tabela()
#RLC('C:/Users/Gabriel/Downloads/P3/RLC/RLC', scale='log', R=300+21*5, L=5e-3, C=4e-7).tabela()
