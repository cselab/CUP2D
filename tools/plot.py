import pandas
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.colors as mc
import matplotlib.collections as mcol
from matplotlib.legend_handler import HandlerLineCollection
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import colorsys
import numpy as np
from progress.bar import Bar
import pickle, pprint
import argparse
import seaborn as sns
sns.set_theme()

################################## UTILS ##################################
colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6']

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], min(1, amount * c[1]), c[2])

class HandlerDashedLines(HandlerLineCollection):
    """
    Custom Handler for LineCollection instances.
    """
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # figure out how many lines there are
        numlines = len(orig_handle.get_segments())
        xdata, xdata_marker = self.get_xdata(legend, xdescent, ydescent,
                                             width, height, fontsize)
        leglines = []
        # divide the vertical space where the lines will go
        # into equal parts based on the number of lines
        ydata = np.full_like(xdata, height / (numlines + 1))
        # for each line, create the line at the proper location
        # and set the dash pattern
        for i in range(numlines):
            legline = Line2D(xdata, ydata * (numlines - i) - ydescent)
            self.update_prop(legline, orig_handle, legend)
            # set color, dash pattern, and linewidth to that
            # of the lines in linecollection
            try:
                color = orig_handle.get_colors()[i]
            except IndexError:
                color = orig_handle.get_colors()[0]
            try:
                dashes = orig_handle.get_dashes()[i]
            except IndexError:
                dashes = orig_handle.get_dashes()[0]
            try:
                lw = orig_handle.get_linewidths()[i]
            except IndexError:
                lw = orig_handle.get_linewidths()[0]
            if dashes[0] is not None:
                legline.set_dashes(dashes[1])
            legline.set_color(color)
            legline.set_transform(trans)
            legline.set_linewidth(lw)
            leglines.append(legline)
        return leglines
###########################################################################

def plotDragTime( root, runname, speed, radius, i ):
  data    = np.loadtxt(root+runname+"/forceValues_0.dat", skiprows=1)
  time    = data[:,0]*speed/radius
  totDrag = data[:,1]/(radius*speed*speed)

  #### uncomment if you want to plot pressure/viscous drag separately ####
  #presDrag = -data[:,3] / (radius*speed*speed)
  #viscDrag = -data[:,5] / (radius*speed*speed)
  #plt.plot(time, presDrag, color=lighten_color(colors[i],1.2))# , label=runname+", $C_p$")
  #plt.plot(time, viscDrag, color=lighten_color(colors[i],1.4))# , label=runname+", $C_v$")
  ########################################################################

  #### uncomment and adapt i to levelMax for which you want to plot the result ####
  # if i == 5:
  #   plt.plot(time, totDrag,  color=lighten_color(colors[i],1), label="present (9 levels)" )# , label=runname+", $C_D")
  #################################################################################

  # for disk
  # plt.plot(time, totDrag,  color=lighten_color(colors[i],1), label=runname)
  # for naca
  plt.plot(time, totDrag,  color=lighten_color("green",0.25+i/4), label="$\\alpha$={:0d}".format(i))
  plt.grid()
  
def plotLiftTime( root, runname, speed, radius, i ):
  data    = np.loadtxt(root+runname+"/forceValues_0.dat", skiprows=1)
  time    = data[:,0]*speed/radius
  totDrag = data[:,2]/(radius*speed*speed)

  # all angles
  # plt.plot(time, totDrag,  color=lighten_color("blue",0.25+i/20), label=runname)
  # before shedding i=0,..,6
  plt.plot(time, totDrag,  color=lighten_color("blue",0.25+i/4), label="$\\alpha$={:0d}".format(i))
  plt.grid()

def plotDragTimeCylinder():
  cases = [ "Re40", "Re550", "Re3000", "Re9500", "Re40000", "Re100000"]
  # cfl = ["0.02", "0.06", "0.2"]
  for case in cases:
    # case   = "Re100000"
    cfl    = "0.06"
    levels = "04" 

    if case != "Re100000":
      validationPath = "/project/s929/pweber/diskValidationData/"+case+".txt"
      data = np.loadtxt(validationPath, delimiter=",")

    rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/CUPamr/"
    rootPROJECT = "/project/s929/pweber/CUP2Damr/disk/"

    runname = ["disk"+case+"_levels04_cfl"+cfl,"disk"+case+"_levels05_cfl"+cfl,"disk"+case+"_levels06_cfl"+cfl, "disk"+case+"_levels07_cfl"+cfl, "disk"+case+"_levels08_cfl"+cfl, "disk"+case+"_levels09_cfl"+cfl ]
    # runname = ["disk"+case+"_levels"+levels+"_cfl0.02", "disk"+case+"_levels"+levels+"_cfl0.06", "disk"+case+"_levels"+levels+"_cfl0.2" ]

    speed = 0.2
    radius = 0.1
    for i in range( len(runname) ):
      plotDragTime( rootPROJECT, runname[i], speed, radius, i )

    if case == "Re40000":
      plt.plot(data[:,0], data[:,1], "--k", label="Rossinelli et al. (2015)")
    elif case != "Re100000":
      plt.plot(data[:,0], data[:,1], "2k", label="Koumoutsakos et al. (1995)")


    plt.ylim([0,8])
    plt.xlim([0,10])
    plt.xlabel("Time $T=tu_\infty/r$")
    plt.ylabel("Drag Coefficient $C_D=2|F_x|/cu_\infty^2$")
    plt.grid()
    plt.legend()
    plt.show()

def plotDragTimeNaca():
  levels = "05"
  case = np.arange(7)

  # validationPath = "/project/s929/pweber/nacaValidationData/"+case+".txt"
  # data = np.loadtxt(validationPath, delimiter=",")

  rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
  rootPROJECT = "/project/s929/pweber/CUP2Damr/naca/"

  runname = [ "nacaRe1000_levels"+levels+"_angle{:02d}".format(a) for a in case ]

  speed = 0.2
  charlength = 0.1 #1/2 chordlength
  for i in range( len(runname) ):
    plotDragTime( rootSCRATCH, runname[i], speed, charlength, i )

  plt.ylim([0,0.2])
  plt.xlim([0,100])
  plt.xlabel("Time $T=2tu_\infty/c$")
  plt.ylabel("Drag Coefficient $C_D=2|F_x|/cu_\infty^2$")
  plt.grid()
  plt.legend()
  plt.show()

def plotLiftTimeNaca():
  levels = "05"
  case = np.arange(7)

  # validationPath = "/project/s929/pweber/nacaValidationData/"+case+".txt"
  # data = np.loadtxt(validationPath, delimiter=",")

  rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
  rootPROJECT = "/project/s929/pweber/CUP2Damr/naca/"

  runname = [ "nacaRe1000_levels"+levels+"_angle{:02d}".format(a) for a in case ]

  speed = 0.2
  charlength = 0.1 #1/2 chordlength
  for i in range( len(runname) ):
    plotLiftTime( rootSCRATCH, runname[i], speed, charlength, i )

  plt.ylim([0,0.5])
  plt.xlim([0,100])
  plt.xlabel("Time $T=2tu_\infty/c$")
  plt.ylabel("Lift Coefficient $C_L=2|F_y|/cu_\infty^2$")
  plt.grid()
  plt.legend()
  plt.show()

def plotDragLiftTimeNaca():
  levels = "07"
  # case = np.arange(7)
  # case = np.arange(7,14)
  case = np.arange(31)

  # validationPath = "/project/s929/pweber/nacaValidationData/"+case+".txt"
  # data = np.loadtxt(validationPath, delimiter=",")

  rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
  rootPROJECT = "/project/s929/pweber/CUP2Damr/naca/"

  runname = [ "nacaRe1000_levels"+levels+"_angle{:02d}".format(a) for a in case ]

  speed = 0.2
  charlength = 0.1 #1/2 chordlength
  for i in range( len(runname) ):
    plotDragTime( rootSCRATCH, runname[i], speed, charlength, i )
    plotLiftTime( rootSCRATCH, runname[i], speed, charlength, i )

  plt.ylim([-1,1])
  plt.xlim([0,40])
  plt.xlabel("Time $T=2tu_\infty/c$")
  plt.ylabel("Force Coefficients $C_{D/L}=2F_{x/y}/cu_\infty^2$")
  plt.grid(b=True, which='major', color="white", linestyle='-')
  plt.tight_layout()
  # plt.legend()
  plt.show()

def plotDragAngleNaca():
  levels = "07"
  case = np.arange(31)

  validationPath = "/project/s929/pweber/nacaValidationData/Re1000.txt"
  validData = np.loadtxt(validationPath, delimiter=",")

  rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
  rootPROJECT = "/project/s929/pweber/CUP2Damr/naca/"

  runname = [ "nacaRe1000_levels"+levels+"_angle{:02d}".format(a) for a in case ]

  speed = 0.2
  charlength = 0.1 #1/2 chordlength

  quotient = []
  for run in runname:
    data = np.loadtxt(rootSCRATCH+run+"/forceValues_0.dat", skiprows=1)
    time = data[:,0]*speed/charlength
    drag = data[:,1]/(charlength*speed*speed)
    lift = -data[:,2]/(charlength*speed*speed)
    quotient.append( np.mean(lift[-40000:]) / np.mean(drag[-40000:]) )

  plt.plot( case, quotient, label="present")
  plt.plot( validData[:,0], validData[:,1], "2k", label="Kurtulus (2016)")
  plt.xlabel("Angle $\\alpha$")
  plt.ylabel("Force Coefficient $C_L/C_D$")
  plt.grid(b=True, which='major', color="white", linestyle='-')
  plt.tight_layout()
  # plt.legend()
  plt.show()

def plotSwimmerSpeed():
  # Helper function to rotate
  def rotate( x, y, theta ):
    xRot = x*np.cos(-theta) - y*np.sin(-theta)
    yRot = x*np.sin(-theta) + y*np.cos(-theta)
    return xRot, yRot

  case = ["Re100", "Re158", "Re251", "Re398", "Re631", "Re1000", "Re1585", "Re2512", "Re3981", "Re6310", "Re10000", "Re100000"]
  # case = ["Re100"]
  levels = "05"
  L = 0.2
  rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
  rootPROJECT = "/project/s929/pweber/CUP2Damr/naca/"

  runname = [ "stefanFish{}_levels".format(Re)+levels for Re in case ]

  for i, run in enumerate(runname):
    data = np.loadtxt(rootSCRATCH+run+"/velocity_0.dat", skiprows=1)
    time = data[:,0]
    angle= data[:,6]
    u    = data[:,7] / L
    v    = data[:,8] / L

    # compute average quantities (in [T/2,..,])
    indices = (time>1/2) & (time<25-1/2)
    timeAv  = time[indices]
        # plot rawdata
    plt.plot( timeAv, u[indices], color="blue" , linestyle="--", alpha=0.2 )
    plt.plot( timeAv, v[indices], color="green", linestyle="--", alpha=0.2 )
    uAv     = []
    vAv     = []
    angleAv = []
    for t in timeAv:
      indices = (time>t-1/2) & (time<t+1/2)
      angleAverage = np.mean( angle[indices] )
      uAverage = np.mean( u[indices] )
      vAverage = np.mean( v[indices] )
      uAvRot, vAvRot = rotate( uAverage, vAverage, angleAverage )
      angleAv.append( angleAverage )
      uAv.append( uAvRot )
      vAv.append( vAvRot )

    plt.plot(timeAv, uAv, color="blue" , label="$v_{\parallel}$")
    plt.plot(timeAv, vAv, color="green", label="$v_{\perp}$")
    # plt.plot(timeAv, angleAv)


    # plt.plot( time, angle )
    plt.xlabel("Time $t$")
    plt.ylabel("Velocity $v/(L/T)$")
    plt.grid(b=True, which='major', color="white", linestyle='-')
    plt.tight_layout()
    plt.title(case[i])
    # plt.legend()
    plt.show()

def plotSwimmerScaling():
  # Helper function to rotate
  def rotate( x, y, theta ):
    xRot = x*np.cos(-theta) - y*np.sin(-theta)
    yRot = x*np.sin(-theta) + y*np.cos(-theta)
    return xRot, yRot

  # Re=10 <-> NU=0.004; Re=100 <-> NU=0.0004; Re=158 <-> NU=0.000253165; Re=251 <-> NU=0.000159363; Re=398 <-> NU=0.000100503; Re=631 <-> NU=0.0000633914; Re=1000 <-> NU=0.00004; Re=1585 <-> NU=0.0000252366; Re=2512 <-> NU=0.0000159236; Re=3981 <-> NU=0.0000100477; Re=6310 <-> NU=0.00000633914; Re=10'000 <-> NU=0.000004; Re=100'000 <-> NU=0.0000004
  case = ["Re100", "Re158", "Re251", "Re398", "Re631", "Re1000", "Re1585", "Re2512", "Re3981", "Re6310", "Re10000"] #, "Re100000"]
  nu = [0.0004, 0.000253165, 0.000159363, 0.000100503, 0.0000633914, 0.00004, 0.0000252366, 0.0000159236, 0.0000100477, 0.00000633914, 0.000004] #, 0.0000004]
  L = 0.2
  omega = 2*np.pi
  A = 5.7*L
  # case = ["Re100"]
  levels = "06"

  rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
  rootPROJECT = "/project/s929/pweber/CUP2Damr/naca/"

  runname = [ "stefanFish{}_levels".format(Re)+levels for Re in case ]

  Re = []
  Sw = []
  for i, run in enumerate(runname):
    data = np.loadtxt(rootSCRATCH+run+"/velocity_0.dat", skiprows=1)
    time = data[:,0]
    angle= data[:,6]
    u    = data[:,7]
    v    = data[:,8]

    # compute average quantities (in [T/2,..,])
    indices = (time>1/2) & (time<25-1/2)
    timeAv  = time[indices]
    

    t = timeAv[-1]
    indices = (time>-1/2) & (time<t+1/2)
    uAverage = np.mean( u[indices] )
    vAverage = np.mean( v[indices] )
    U = np.sqrt( uAverage**2 + vAverage**2 )

    ReEff = U*L/nu[i]
    SwEff = omega*A*L/nu[i]
    
    Re.append(ReEff)
    Sw.append(SwEff)


  SwValues = 10**np.arange(3,7)
  plt.plot( SwValues, SwValues/70,     color="red",   linewidth=1, linestyle="-", label="$Sw\sim Re$") 
  plt.plot( SwValues, SwValues**(4/3)/3500, color="black", linewidth=1, linestyle="-", label="$Sw\sim Re^{4/3}$") 
  plt.plot( Sw, Re, "ko" )
  plt.xscale("log")
  plt.yscale("log")
  plt.xlabel("Sw")
  plt.ylabel("Re")
  plt.grid(b=True, which='major', color="white", linestyle='-')
  plt.tight_layout()
  plt.legend()
  plt.show()

if __name__ == '__main__':
  # plotDragLift()
  # plotDragTimeCylinder()
  # plotDragTimeNaca()
  # plotLiftTimeNaca()
  # plotDragLiftTimeNaca()
  # plotDragAngleNaca()
  # plotSwimmerSpeed( )
  plotSwimmerScaling()
