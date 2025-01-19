import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as sc

class Fisica:
    def __init__(self, arq=None, data=None, R1=None, C1=None):
        self.R1 = R1 # Constantes iniciais
        self.C1 = C1
        if isinstance(data,pd.DataFrame): # Importante para possibilitar a clonagem
            self.data = data
        else:
            data = pd.read_csv(rf'C:\Users\Gabriel\Downloads\{arq}',sep='\t') # Altere o diretório para aquele com as pastas conf1, conf2, etc
            data.loc[-1] = data.columns  # Adiciona o cabeçalho atual como uma nova linha
            data.index = data.index + 1  # Ajusta os índices para deslocar as linhas
            df = data.sort_index()     # Reordena as linhas pelo índice

            # Definindo novos nomes de cabeçalho
            df.columns = ['t', 'A1']
            df['t'] = df['t'].apply(lambda x: float(x))
            df['A1'] = df['A1'].apply(lambda x: float(x))
            self.data = df

    def lim_sup(self, lim): # Clona a entidade com um teto
        return Fisica(data=self.data[self.data['t'] <= lim], R1=self.R1, C1=self.C1)
    
    def lim_inf(self, lim): # Clona a entidade com um piso
        return Fisica(data=self.data[lim <= self.data['t']], R1=self.R1, C1=self.C1)
    
    def discr(self, cons): # Discrepância
        if self.R1 != None and self.C1 != None:
            print('Discrepância do capacitor de seu val. nominal: ', 100*(cons/self.R1-self.C1)/self.C1, ' %')

    def RC_ajuste(self): # Calcula tau pelo ajuste e plota a curva
        f = lambda x, a, t0: (1-np.e**(-(x-t0)/a))*5
        coef = sc.curve_fit(f,self.data['t'],self.data['A1'],p0=[self.R1*self.C1,0])
        print('Constante de tempo pelo ajuste: ', coef[0][0])
        self.discr(coef[0][0])

        self.pre_plot(self.data['t'], f(self.data['t'], *coef[0]), 'red', 2)

    def bruto(self, color='blue'): # Plota os dados brutos
        self.pre_plot(list(self.data['t']),list(self.data['A1']), color, 3)

    def VR(self): # Plota VR(t)
        self.pre_plot(self.data['t'], list(5-self.data['A1']), 'green', 1)

    def RC_1e(self): # Calcula tau por 1/e
        for k in range(len(self.data['t'])):
            if list(self.data['A1'])[k] <= (1-1/np.e)*5 <= list(self.data['A1'])[k+1]:
                val = list(self.data['t'])[k] + (list(self.data['t'])[k+1] - list(self.data['t'])[k])*((1-1/np.e)*5 - list(self.data['A1'])[k])/(list(self.data['A1'])[k+1] - list(self.data['A1'])[k])
                print('simples medida de (1/e): ', val)
                self.discr(val)
                break
    
    def pre_plot(self, X, Y, color, esp=4): # Método para plotagem
        plt.ylabel('Voltagem (V)')
        plt.xlabel('Tempo (seg)')
        plt.plot(X, Y, color=color, linewidth=esp)





if __name__ == '__main__':
    # Prática 2, configuração 1
    qua = Fisica('conf1/CH0.txt').lim_sup(3/(2*61)) # Vá alterando "conf1" pelos demais, para ver para as outras configurações. Aqui e na linha abaixo
    cap = Fisica('conf1/CH1.txt', R1=1e4, C1=1e-7).lim_sup(3/(2*61))

    qua.bruto()
    cap.bruto('orange')
    cap.VR()
    cap = cap.lim_sup(1/(2*61))
    cap.RC_ajuste()
    cap.RC_1e()
    plt.show()
