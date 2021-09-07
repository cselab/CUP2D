import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
import numpy as np
import scipy as sp
from scipy import integrate

colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

def lighten_color(color, amount=0.5):
  try:
      c = mc.cnames[color]
  except:
      c = color
  c = colorsys.rgb_to_hls(*mc.to_rgb(c))
  return colorsys.hls_to_rgb(c[0], min(1, amount * c[1]), c[2])

# "The initial flow past an impulsively started circular cylinder", W. M. Collins and S. C. R. Dennis (1973)
def dragCollinsDennis( Re, t, i=0 ):
  k = 2*np.sqrt(2*t/Re)
  fricDrag = np.pi*(1/np.sqrt(2*Re*t))*(2.257+k-0.141*k**2+0.062*k**3 - (0.092-1.6*k+6.9*k**2-22.7*k**3)*t**2 - (0.074-1.24*k+12.12*k**2+24.35*k**3)*t**4 + (0.008+0.196*k)*t**6 )
  presDrag = np.pi*(1/np.sqrt(2*Re*t))*(2.257+k-0.141*k**2+(4.59-22.1*k+78.8*k**2)*t**2 + (2.68-40.5*k+219.3*k**3)*t**4 + 0.894*t**6 )
  if i == 1:
    return fricDrag
  if i == 2:
    return presDrag
  return fricDrag+presDrag


def plotForceTime( root, runname, Re, i):
  u = 0.2
  r = 0.1
  data = np.loadtxt(root+runname+"/forceValues_0.dat", skiprows=1)
  #t  = data[:,0] * u/r
  #Cd = data[:,1] / (r*u*u)
  #Cp = data[:,3] / (r*u*u)
  #Cv = data[:,5] / (r*u*u)
  #err1 = np.abs(Cv[1::] - dragCollinsDennis(Re,t[1::],1))
  #err2 = np.abs(Cp[1::] - dragCollinsDennis(Re,t[1::],2))
  #print(runname,sp.integrate.simps(err1, t[1::]),sp.integrate.simps(err2, t[1::]))
  #plt.plot(t, Cv, color=lighten_color(colors[i],1.), label=runname)
  #print(Cv)
  #plt.plot(t, Cp, color=lighten_color(colors[i+1],1.), label=runname)
  #plt.plot(t, Cd, color=lighten_color(colors[i+2],1), label=runname)
  plt.plot(data[:,0] * (u/r), data[:,1] / (r*u*u), color=lighten_color(colors[i],1.), label=runname)
  plt.plot(data[:,0] * (u/r), data[:,3] / (r*u*u), color=lighten_color(colors[i],1.), label=runname)
  plt.plot(data[:,0] * (u/r), data[:,5] / (r*u*u), color=lighten_color(colors[i],1.), label=runname)
  
def plotDragTimeCylinder():

  Re = 10000
  root = "/scratch/snx3000/mchatzim/CUP2D/"
  runname = []
  runname.append("Re10k")

  t = np.linspace(1e-10,0.5,1001)
  plt.plot( t, dragCollinsDennis( Re, t, 0), linestyle="--", color="black", label="Collins and Dennis (1973)")
  plt.plot( t, dragCollinsDennis( Re, t, 1), linestyle="--", color="black", label="Collins and Dennis (1973)")
  plt.plot( t, dragCollinsDennis( Re, t, 2), linestyle="--", color="black", label="Collins and Dennis (1973)")

  for i in range( len(runname) ):
    plotForceTime( root, runname[i], Re, i)

  plt.xlim([1e-6,0.5])
  #plt.ylim([1e-6,5])
  plt.ylim([1e-2,1.0])
  plt.xlabel("Time $T=tu_\infty/r$")
  plt.ylabel("Drag Coefficient $C_D=|F_x|/ru_\infty^2$")
  plt.legend(loc = 'upper right')
  #plt.ylim([1e-6,1e4])
  #plt.xscale("log")
  #plt.yscale("log")
  plt.grid()
  plt.show()

if __name__ == '__main__':
  plotDragTimeCylinder()
