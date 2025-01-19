# Prática 3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Fisica2:
    def __init__(self, arq=None, data=None, R1=None, L1=None):
        self.R1 = R1
        self.L1 = L1
        if isinstance(data,pd.DataFrame):
            self.data = data
        else:
            data = pd.read_csv(rf'C:\Users\Gabriel\Downloads\{arq}')
            data['Time(s)'] = data['Time(s)'].apply(lambda x: float(x))
            data['Time(s)'] -= min(data['Time(s)'])
            data['CH1V'] = data['CH1V'].apply(lambda x: float(x))
            data['CH1V'] -= min(data['CH1V'])
            data['CH2V'] = data['CH2V'].apply(lambda x: float(x))
            data['CH2V'] -= min(data['CH2V'])
            self.data = data

    def lim_sup(self, lim):
        return Fisica2(data=self.data[self.data['Time(s)'] <= lim], R1=self.R1, L1=self.L1)
    
    def lim_inf(self, lim):
        return Fisica2(data=self.data[lim <= self.data['Time(s)']], R1=self.R1, L1=self.L1)
    
    def discr(self, cons):
        if self.R1 != None and self.L1 != None:
            print('Discrepância do indutor de seu val. nominal: ', 100*(self.R1*cons-self.L1)/self.L1, ' %')

    def RC_ajuste(self):
        import scipy.optimize as sc
        f = lambda x, a, t0, A0: A0*(1-np.e**(-(x-t0)/a))
        coef = sc.curve_fit(f, self.data['Time(s)'], self.data['CH2V'], p0=[self.L1/self.R1, 0, 1])
        print('Constante de tempo pelo ajuste: ', coef[0][0])
        self.discr(coef[0][0])

        self.pre_plot(self.data['Time(s)'], f(self.data['Time(s)'], *coef[0]), 'red', 2)

    def bruto(self, color='blue', ch='CH1V'):
        self.pre_plot(list(self.data['Time(s)']),list(self.data[ch]), color, 3)

    def VL(self):
        self.pre_plot(self.data['Time(s)'], list(5-self.data['CH2V']), 'green', 1)

    def RC_1e(self):
        for k in range(len(self.data['Time(s)'])-1):
            if list(self.data['CH2V'])[k] <= (1-1/np.e) <= list(self.data['CH2V'])[k+1]:
                val = list(self.data['Time(s)'])[k] + (list(self.data['Time(s)'])[k+1] - list(self.data['Time(s)'])[k])*((1-1/np.e)*5 - list(self.data['CH2V'])[k])/(list(self.data['CH2V'])[k+1] - list(self.data['CH2V'])[k])
                print('simples medida de (1/e): ', val)
                self.discr(val)
                break

    def imped(self):
        print(f'Z = {self.R1} + j{16*1e3*self.L1/np.pi}')
    
    def pre_plot(self, X, Y, color, esp=4):
        plt.ylabel('Voltagem (V)')
        plt.xlabel('Tempo (seg)')
        plt.plot(X, Y, color=color, linewidth=esp)


if __name__ == '__main__':
    cap = Fisica2('RL/RigolDS0.csv', R1=1e2+5*21.352100127683098, L1=5e-3).lim_sup(3/(2*1e3)) # Temos no outro caso: R1=1e2+3*21.352100127683098, L1=3e-3 e 'RL/3mh/RigolDSd1.csv'

    cap.bruto(ch='CH1V')
    cap.bruto('orange', ch='CH2V')
    cap.VL()
    cap = cap.lim_sup(1/(2*1e3))
    cap.RC_ajuste()
    cap.RC_1e()
    cap.imped()
    plt.show()
