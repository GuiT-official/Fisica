# Prática 2C
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sc

class Fisica3:
    def __init__(self, arq=None, data=None, om=None, alp=None):
        self.om = om
        self.alp = alp
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
        return Fisica3(data=self.data[self.data['Time(s)'] <= lim], om=self.om, alp=self.alp)
    
    def lim_inf(self, lim):
        return Fisica3(data=self.data[lim <= self.data['Time(s)']], om=self.om, alp=self.alp)
    
    def discr(self, consw, consa):
        pass
        if self.om != None and self.alp != None:
            print('Discrepância de omega de seu val. nominal: ', 100*(consw-self.om)/self.om, ' %')
            print('Discrepância de alpha de seu val. nominal: ', 100*(consa-self.alp)/self.alp, ' %')

    def RC_ajuste(self,t=0):
        f = lambda x, Voff, V0, alp, t0, om: Voff+V0*np.e**(-alp*(x-t0-t))*np.cos(om*(x-t0-t))
        coef = sc.curve_fit(f, self.data['Time(s)'], self.data['CH2V'], p0=[3+1.6/3,1/3,self.alp,25e-5/3,self.om])
        print('Constante omega pelo ajuste: ', coef[0][4])
        print('Constante alpha pelo ajuste: ', coef[0][2])
        self.discr(coef[0][4], coef[0][2])

        self.pre_plot(self.data['Time(s)'], f(self.data['Time(s)'], *coef[0]), 'red', 2)

    def bruto(self, color='blue', ch='CH1V'):
        self.pre_plot(list(self.data['Time(s)']),list(self.data[ch]), color, 3)

    def VL(self):
        self.pre_plot(self.data['Time(s)'], list(5-self.data['CH2V']), 'green', 1)
    
    def pre_plot(self, X, Y, color, esp=4):
        plt.ylabel('Voltagem (V)')
        plt.xlabel('Tempo (seg)')
        plt.plot(X, Y, color=color, linewidth=esp)


if __name__ == '__main__':
    cap = Fisica3('RLC/RigolDSd0000.csv', alp=(21.352100127683098*5+0)/(2*5e-3),om=1/np.sqrt(25e-9*5e-3)).lim_sup(3/(2*1e3)) # Para o de 0

    cap.bruto(ch='CH1V')
    cap.bruto('orange', ch='CH2V')
    cap.VL()
    cap = cap.lim_sup(1/(2*1e3))
    cap.RC_ajuste()
    plt.show()


    cap = Fisica3('RLC/RigolDSd3000.csv', alp=(21.352100127683098*5+300)/(2*5e-3),om=1/np.sqrt(25e-9*5e-3)) # Para o de 300

    cap.bruto(ch='CH1V')
    cap.bruto('orange', ch='CH2V')
    cap.VL()
    cap = cap.lim_sup(2/(2*1e3)).lim_inf(1/(2*1e3))
    cap.RC_ajuste(1/(2*1e3))
    plt.show()


    cap = Fisica3('RLC/RigolDSd6000.csv', alp=(21.352100127683098*5+600)/(2*5e-3),om=1/np.sqrt(25e-9*5e-3)) # Para o de 600

    cap.bruto(ch='CH1V')
    cap.bruto('orange', ch='CH2V')
    cap.VL()
    cap = cap.lim_sup(2/(2*1e3)).lim_inf(1/(2*1e3))
    cap.RC_ajuste(1/(2*1e3))
    plt.show()
