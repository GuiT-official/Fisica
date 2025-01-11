import matplotlib.pyplot as plt
import numpy as np

D = {1510:5.0000,1500:4.9707,1200:3.9687,900:2.9668,600:1.9795,300:0.9824}
err = [np.sqrt((0.05*1500/np.sqrt(5))**2+1/4)]+list(map(lambda x: 0.05*x/np.sqrt(x/300),list(D.keys())[1:]))
plt.errorbar(D.keys(),D.values(),yerr=[4.9e-3]*len(D),xerr=[0.05*1500/np.sqrt(5)+1/4]+list(map(lambda x: 0.05*x/np.sqrt(x/300),list(D.keys())[1:])),fmt='o',markersize=6,capsize=5,ecolor='green')

X = np.linspace(0,1510)
f = lambda x, xer, yer: (D[1500]+yer)/(1500+xer)*x
plt.plot(X, f(X,err[list(D.keys()).index(1500)],-4.9e-3), label='Linha até o (1500+eps, VA1-eps)', color='red')
plt.plot(X, f(X,-err[list(D.keys()).index(1500)],4.9e-3), label='Linha até o (1500-eps, VA1+eps)', color='red')
plt.legend()

plt.xlabel('Resistência (Ohms)')
plt.ylabel('Tensão (Volts)')
plt.show()
