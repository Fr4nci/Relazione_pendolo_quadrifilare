import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
data = np.loadtxt("test2_600.txt", unpack=True)
# Modelli teorici usati
def periodo_taylor(x, t0):
    return(t0 * (1 + (1./16.) * x ** 2))
def periodo_taylor_2(x, t0):
    return(t0 * (1 + (1./16.) * x ** 2 + (11./3072.) * x ** 4))
def velocita(x, v_0, _lamdba):
    return(v_0 * np.e ** (-_lamdba * x))

# Grafico Periodo - Tempo
tempo = data[0]
periodo = data[1]
transit_time = data[2]

w = 2.05 * 10 ** (-2)
err_w = 0.005 * 10 ** (-2)
l = 1.15
err_l = 0.001
d = 1.20
err_d = 0.001
g = 9.81
err_trans_time = np.full(shape=len(transit_time), fill_value=(4 * 10 ** (-6))/np.sqrt(12), dtype=float)
vel = (w/transit_time) * (l/d)
err_vel = np.sqrt((err_w/w) ** 2 + (err_l/l) ** 2 + (err_d/d)**2 + (err_trans_time/transit_time) ** 2) * vel
plt.plot(tempo, periodo)
plt.savefig("Grafico_periodo_tempo.pdf")
plt.show()
plt.plot(tempo, vel)
plt.savefig("Grafico_velocita_tempo.pdf")
plt.show()
popt, pcov = curve_fit(velocita, tempo, vel, sigma=err_vel, p0=[0.58, 0.01/5])
v_0_hat, _lambda_hat = popt
sigma_v_0, sigma_lamdba = np.sqrt(np.diag(pcov))
fig = plt.figure("Fit e residui velocità")
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.5))
print(f"v_0: {v_0_hat} +- {sigma_v_0}, lambda: {_lambda_hat} +- {sigma_lamdba}")
ax1.plot(tempo, velocita(tempo, v_0_hat, _lambda_hat), color='orange', label="Modello di best-fit")
ax1.errorbar(tempo, vel, yerr=err_vel, xerr=err_trans_time, fmt='o', label="Dati")
ax1.set_xlabel("t [s]")
ax1.set_ylabel("v [m/s]")
ax1.grid(color='lightgray', ls="dashed")
ax1.legend()
# Residui grafico velocità tempo
acc = 0
for el in range(len(vel)):
    acc = acc + ((vel[el] - velocita(tempo[el], v_0_hat, _lambda_hat))/err_vel[el]) ** 2
print(acc)
print(len(vel))
residui = vel - velocita(tempo, v_0_hat, _lambda_hat)
ax2.errorbar(tempo, residui, yerr=err_vel, fmt='o')
ax2.axhline(y=0, color="orange")
ax2.set_xlabel("t [s]")
ax2.set_ylabel("Residui [m/s]")
ax2.grid(color='lightgray', ls='dashed')
plt.savefig("Fit_velocita.pdf")
plt.show()
# Sezione Taylor con due ordini di espansione differenti
data = np.loadtxt("test4_angolo_grand_1.txt", unpack=True)
tempo = data[0]
periodo = data[1]
transit_time = data[2]
k = (w ** 2 * l)/(2 * (transit_time ** 2) * (d ** 2) * g)
sigma_k = np.sqrt(2*(err_w/w) ** 2 + (err_l/l)** 2 + 2 * (err_trans_time/transit_time) ** 2 + 2 * (err_d/d) ** 2) * k
theta_0 = np.arccos(1 - k)
err_theta_0 = (1/np.sqrt(1-(1-k)**2)) * sigma_k
err_periodo = np.full(shape=len(periodo), fill_value=0.0001, dtype=float)
popt, pcov = curve_fit(periodo_taylor, theta_0, periodo, sigma=err_periodo)
t0_hat = popt
sigma_t0 = np.sqrt(np.diag(pcov))
popt, pcov = curve_fit(periodo_taylor_2, theta_0, periodo, sigma=err_periodo)
t1_hat = popt
sigma_t1 = np.sqrt(np.diag(pcov))
plt.errorbar(theta_0, periodo, yerr=err_periodo, xerr=err_theta_0, fmt='o')
x = np.linspace(0, np.pi/2)
plt.plot(x, periodo_taylor(x, t0_hat))
plt.plot(x, periodo_taylor_2(x, t1_hat))
plt.savefig("Fit_ampiezza.pdf")
plt.show()
print(f"{t0_hat} +- {sigma_t0}")
print(f"{t1_hat} +- {sigma_t1}")
residui1 = periodo - periodo_taylor(theta_0, t0_hat)
residui2 = periodo - periodo_taylor_2(theta_0, t1_hat)
plt.errorbar(theta_0, residui1, yerr=err_periodo, xerr=err_theta_0, fmt='o', color="blue")
plt.axhline(y=0, color="orange")
plt.savefig("Residui_taylor_2.pdf")
acc = 0
for el in range(len(periodo)):
    acc = acc + ((periodo[el] - periodo_taylor(theta_0[el], t0_hat))/err_periodo[el]) ** 2
print(f"Il chi quadro di Taylor al secondo ordine è: {acc}")
plt.show()
plt.errorbar(theta_0, residui2, yerr=err_periodo, xerr=err_theta_0, fmt='o', color="blue")
plt.axhline(y=0, color="orange")
plt.savefig("Residui_taylor_4.pdf")
plt.show()
acc = 0
for el in range(len(periodo)):
    acc = acc + ((periodo[el] - periodo_taylor_2(theta_0[el], t1_hat))/err_periodo[el]) ** 2
print(f"Il chi quadro di Taylor al quarto ordine è: {acc}")