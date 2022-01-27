import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme()
sns.set_style("whitegrid")

#### load data
pathSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
runname = [ f"KolmogorovFlow_N={N}/EnergySpectrum_N={N}.out" for N in 2**np.arange(5,10) ] #11

for i, run in enumerate(runname):
	data = np.loadtxt(pathSCRATCH+run)
	freq = data[0,:]
	energy = data[1,:] / (2**(6+i)+1)**2
	plt.loglog(freq, energy, label="N={}".format(2**(5+i)))

wavenumbers = np.arange(0, 100, 0.15915494309189535)
# plt.loglog(wavenumbers, 7*10**-4*wavenumbers**(-5/3), 'k--', label="$\\propto k^{-5/3}$") 
plt.loglog(wavenumbers, 5*10**-4*wavenumbers**(-5)*np.log(wavenumbers*2*np.pi)**(-1/3), 'k--', label="$\\propto k^{-5}$")

plt.legend()
plt.xlabel("Wavenumber $k$")
plt.ylabel("Energy $E(k)$")
plt.legend()
plt.rcParams["figure.figsize"] = (12,4)
plt.tight_layout()
# plt.show()
plt.savefig("KFspectrum.eps", dpi=300)