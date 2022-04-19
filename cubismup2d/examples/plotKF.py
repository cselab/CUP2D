import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
# sns.set_theme()
# sns.set_style("whitegrid")
# sns.set(rc={"xtick.minor.visible" : True, "ytick.minor.visible" : True})

Nsmall = 64

#### load data
# pathSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
pathSCRATCH = "./"
# runname = [ f"KolmogorovFlow_N={N}/EnergySpectrum_N={N}.out" for N in 2**np.arange(5,10) ] #11
runname = [ f"EnergySpectrum_N={N}_Cs=0.0.out" for N in [Nsmall, 128] ] #11
runname.append(f"EnergySpectrum_N={Nsmall}_Cs=0.2.out")

for i, run in enumerate(runname):
	data = np.loadtxt(pathSCRATCH+run)
	freq = data[0,:]
	if (i==0) or (i == 2): 
		energy = data[1,:] / Nsmall**2
	else:
		energy = data[1,:] / 128**2
	# energy = data[1,:] / (2**(6+i)+1)**2
	# plt.loglog(freq, energy, label="N={}".format(2**(5+i)))
	plt.loglog(freq, energy, label="{}".format(run))

wavenumbers = np.arange(0, 10, 0.15915494309189535)
# plt.loglog(wavenumbers, 7*10**-4*wavenumbers**(-5/3), 'k--', label="$\\propto k^{-5/3}$") 
# plt.loglog(wavenumbers, 5*10**-4*wavenumbers**(-4)*np.log(wavenumbers*2*np.pi)**(-1/3), 'k--', label="$\\propto k^{-5}$")
plt.loglog(wavenumbers, 5*10**-2*wavenumbers**(-4), 'k--', label="$\\propto k^{-4}$")
plt.tick_params(axis='both', which='minor')
plt.xlabel("Wavenumber $k$")
plt.ylabel("Energy $E(k)$")
plt.legend()
plt.rcParams["figure.figsize"] = (12,4)
plt.tight_layout()
plt.show()
# plt.savefig("KFspectrum.eps", dpi=300)