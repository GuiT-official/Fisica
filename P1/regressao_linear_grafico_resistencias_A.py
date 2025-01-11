import matplotlib.pyplot as plt
import numpy as np

D = {1510:5.0000,1500:4.9707,1200:3.9687,900:2.9668,600:1.9795,300:0.9824}
plt.scatter(D.keys(),D.values())

coef = sum(D[k]*k for k in D.keys())/sum(k**2 for k in D.keys())
print(coef)

incer_corr = np.sqrt(sum(coef*k-D[k] for k in D.keys())/((len(D)-2)*sum((k-sum(D.keys())/len(D))**2 for k in D.keys())))
print('Incerteza: ', incer_corr)

X = np.linspace(0,1510)
f = lambda x: coef*x
plt.plot(X, f(X), label=f'f(x)={coef}x', color='red')
plt.legend()

plt.xlabel('Resistência (Ohms)')
plt.ylabel('Tensão (Volts)')

for k in range(len(D)-1):
    r = (((list(D.values())) + [0])[1:][k]-(list(D.values()) + [0])[1:][k+1])/coef
    print(f'Resistência ajustada para o resistor de 300 de n° {k}: ', r)
    print('Incerteza: ', np.sqrt(2*(0.0049/r)**2 + ((((list(D.values())) + [0])[1:][k]-(list(D.values()) + [0])[1:][k+1])*incer_corr/r**2)**2))

plt.show()
