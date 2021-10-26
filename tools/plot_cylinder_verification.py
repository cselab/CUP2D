import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter

def dragCollinsDennis( Re, t ):
  k = 2*np.sqrt(2*t/Re)
  fricDrag = np.pi*(1/np.sqrt(2*Re*t))*(2.257+k-0.141*k**2+0.062*k**3 - (0.092-1.6*k+6.9*k**2-22.7*k**3)*t**2 - (0.074-1.24*k+12.12*k**2+24.35*k**3)*t**4 + (0.008+0.196*k)*t**6 )
  presDrag = np.pi*(1/np.sqrt(2*Re*t))*(2.257+k-0.141*k**2+(4.59-22.1*k+78.8*k**2)*t**2 + (2.68-40.5*k+219.3*k**3)*t**4 + 0.894*t**6 )
  return presDrag
  
def plotDragTimeCylinder():
  root = "/project/s929/mchatzim/CylinderVerification/"
  cases   = ["550","1000","10000"]
  speed = 0.2
  radius = 0.1
  fig, axs = plt.subplots(len(cases))

  for i in range( len(cases) ):
    data = np.loadtxt(root+"Re"+cases[i]+"/forceValues_0.dat", skiprows=1)
    t = data[:,0] * (speed/radius)
    f = data[:,3]* (1.0/(radius*speed*speed))
    axs[i].plot(t, f, label="Present method")

    t = np.linspace(1e-10,1.0,1000)
    axs[i].plot(t, dragCollinsDennis(int(cases[i]),t), linestyle="--", label="Collins and Dennis (1973)")

    axs[i].set_title("Re="+cases[i],fontsize=10)
    axs[i].set_xlabel("Time",fontsize=8)
    axs[i].set_ylabel("Drag Coefficient",fontsize=8)
    axs[i].set_xlim([0,1])
    axs[i].tick_params(axis='both', which='major', labelsize=8)
    axs[i].set_ylim([0,f[-1]*10])
    axs[i].legend(ncol=2,loc="upper right",prop={'size': 8})
    axs[i].yaxis.set_major_formatter(FormatStrFormatter('%2.1f'))
    axs[i].xaxis.set_major_formatter(FormatStrFormatter('%2.1f'))
  plt.tight_layout()
  plt.savefig("cylinder-verification.eps")

if __name__ == '__main__':
  plotDragTimeCylinder()
