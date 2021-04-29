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

def plotDragTime( root, runname, speed, radius, i, j ):
  ## Helper function to rotate ##
  # def rotate( x, y, theta ):
  #   xRot = x*np.cos(-theta) - y*np.sin(-theta)
  #   yRot = x*np.sin(-theta) + y*np.cos(-theta)
  #   return xRot, yRot
  ################################

  ## load kinematic data ##
  kineticData = np.loadtxt(root+runname+"/velocity_0.dat", skiprows=1)
  kineticTime = kineticData[:,0]
  u = kineticData[:,7]
  v = kineticData[:,8]
  # print("speed:", u, v)
  #########################

  ## load dynamic data ##
  dynamicData = np.loadtxt(root+runname+"/forceValues_0.dat", skiprows=1)
  time    = dynamicData[:,0]
  np.testing.assert_array_equal(time, kineticTime)
  forceX  = dynamicData[:,1]
  forceY  = dynamicData[:,2]
  # print("force:", forceX, forceY)
  #######################

  ## dimensionless time ##
  time    = time*speed/radius
  ########################

  ## compute drag ##
  totDrag = (u*forceX+v*forceY) / np.sqrt(u**2 + v**2)
  dragCoeff = -totDrag / (radius*speed*speed)
  # print("drag:", totDrag, dragCoeff)
  ##################

  #### uncomment if you want to plot pressure/viscous drag separately ####
  # presForceX = dynamicData[:,3]
  # presForceY = dynamicData[:,4]
  # presDrag = u*presForceX+v*presForceY / np.sqrt(u**2 + v**2)
  # presDragCoeff = -presDrag / (radius*speed*speed)

  # viscForceX = dynamicData[:,5]
  # viscForceY = dynamicData[:,6]
  # viscDrag = u*viscForceX+v*viscForceY / np.sqrt(u**2 + v**2)
  # viscDragCoeff = -viscDrag / (radius*speed*speed)
  
  # if i == j+1:
  #   plt.plot(time, presDragCoeff, color=lighten_color(colors[i],1.4), label="$C_p$")
  #   plt.plot(time, viscDragCoeff, color=lighten_color(colors[i],0.6), label="$C_v$")
  ########################################################################


  
  #### to average quantities ####
  # T = 1 #averaging window
  # # restrict time array
  # indices = (time>T/2) & (time<time[-1]-T/2)
  # timeAv  = time[indices]
  # get averages for restricted time array
  # avDrag = []
  # for t in timeAv:
  #   indices  = (time>t-T/2) & (time<t+T/2)
  #   #QoI
  # plt.plot(timeAv, avDrag,  color=lighten_color(colors[i],1), label=runname)
  ######################################

  #### uncomment and adapt i to levelMax for which you want to plot the result ####
  # if i == j+2: 90
  print(time[0], dragCoeff[0])
  if i == 2: 
    index = dragCoeff < 2.0
    print(time[index])
    plt.plot(time[index], dragCoeff[index],  color=lighten_color(colors[i],1), label="$C_D$ present ({} levels)".format(i+4) )# , label=runname+", $C_D")
  #################################################################################

  # for disk
  # plt.plot(time, dragCoeff, color=lighten_color(colors[i],1), label=runname)
  # for naca
  # plt.plot(time, dragCoeff,  color=lighten_color("green",0.25+i/4), label="$\\alpha$={:0d}".format(i))
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
  # "The initial flow past an impulsively started circular cylinder", W. M. Collins and S. C. R. Dennis (1973)
  def dragCollinsDennis( Re, t ):
    k = 2*np.sqrt(2*t/Re)
    fricDrag = np.pi*(1/np.sqrt(2*Re*t))*(2.257+k-0.141*k**2+0.062*k**3 - (0.092-1.6*k+6.9*k**2-22.7*k**3)*t**2 - (0.074-1.24*k+12.12*k**2+24.35*k**3)*t**4 + (0.008+0.196*k)*t**6 )
    presDrag = np.pi*(1/np.sqrt(2*Re*t))*(2.257+k-0.141*k**2+(4.59-22.1*k+78.8*k**2)*t**2 + (2.68-40.5*k+219.3*k**3)*t**4 + 0.894*t**6 )
    # print("diff:", presDrag-fricDrag)
    # equation (80) + (81)
    return fricDrag+presDrag
    # equation (82)
    # return 2*np.sqrt( 8*np.pi/(Re*t) )

  # "Initial flow field over an impulsively started circular cylinder". M. Bar-Lev and H. T. Yang (1975)
  def dragBarLevYang( Re, t ):
    # full expansion from paper
    fricDrag = 2*np.sqrt(np.pi/(Re*t)) + np.pi/Re+15/2*np.sqrt(np.pi*t/Re)/Re - 2*t*np.sqrt(np.pi*t/Re)*(((108*np.sqrt(3)-89)/(30*np.pi)+128/(135*np.pi**2)-11/12)+3*(64/(45*np.pi**2)+(27*np.sqrt(3)-11)/(30*np.pi)-7/12))
    presDrag = 2*np.sqrt(np.pi/(Re*t)) + np.pi/Re*(9-15/np.sqrt(np.pi))+15/2*np.sqrt(np.pi*t/Re)/Re + 2*t*np.sqrt(np.pi*t/Re)*(((108*np.sqrt(3)-89)/(30*np.pi)+128/(135*np.pi**2)-11/12)+3*(64/(45*np.pi**2)+(27*np.sqrt(3)-11)/(30*np.pi)-7/12))
    # print("diff:", presDrag-fricDrag)
    # return tot-drag=fricDrag + presDrag
    return 4*np.sqrt( np.pi/(Re*t) ) + np.pi*(9-15/np.sqrt(np.pi))/Re

  ## for initial time ## "40", "200", "1000", 
  cases = [ "10000" ]
  ######################

  ## for long time ##
  # cases = ["40", "550", "3000", "9500" ]
  ######################

  for j, case in enumerate(cases):
    rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
    # rootPROJECT = "/project/s929/pweber/CUP2Damr/disk/"
    rootPROJECT = "/project/s929/pweber/CUP2Damr/stefanFish/"

    runname = [ "diskRe{}_levels{}".format(case, level) for level in np.arange(4,10) ]
    # runname = ["diskRe"+case+"_levels8_poissonTol9"]
    # runname = [ "diskRe"+case+"_poissonTol11" ]
    # runname = ["disk"+case+"_levels04_cfl"+cfl,"disk"+case+"_levels05_cfl"+cfl,"disk"+case+"_levels06_cfl"+cfl, "disk"+case+"_levels07_cfl"+cfl, "disk"+case+"_levels08_cfl"+cfl, "disk"+case+"_levels09_cfl"+cfl ]
    # runname = ["disk"+case+"_levels"+levels+"_cfl0.02", "disk"+case+"_levels"+levels+"_cfl0.06", "disk"+case+"_levels"+levels+"_cfl0.2" ]
    # "disk"+case+"_levels"+levels+"_poissonTol5", "disk"+case+"_levels"+levels+"_poissonTol6", 
    # runname = ["diskRe"+case+"_levels"+levels+"_poissonTol4", "diskRe"+case+"_levels"+levels+"_poissonTol5", "diskRe"+case+"_levels"+levels+"_poissonTol6", "diskRe"+case+"_levels"+levels+"_poissonTol7", "diskRe"+case+"_levels"+levels+"_poissonTol8", "diskRe"+case+"_levels"+levels+"_poissonTol9" ]
    # "disk"+case+"_levels"+levels+"_poissonTol4", 
    # runname = ["disk"+case+"_levels5_poissonTol"+poisson, "disk"+case+"_levels6_poissonTol"+poisson, "disk"+case+"_levels7_poissonTol"+poisson]
    # runname = ["stefanFishRe1000_levels05"] #, "stefanFishRe1000_levels06", "stefanFishRe1000_levels07"]

    ###### plot validation data ######
    ## for initial time ##
    time = np.linspace(0,0.1,1001)
    plt.plot( time, dragCollinsDennis( int(case), time ), linestyle="--", color="black", label="Collins and Dennis (1973)")
    # plt.plot( time, dragBarLevYang( int(case), time ), linestyle="--", color="lightgrey", label="Bar-Lev and Yang (1975)")
    validationPath = "/project/s929/pweber/diskValidationData/Re"+case+"-start.txt"
    validationData = np.loadtxt(validationPath, delimiter=",")
    plt.plot(validationData[:,0], validationData[:,1], "2k", label="Koumoutsakos and Leonhard (1995)")
    plt.xlim([0.00075543,0.1])
    ######################

    ## for long time ##
    # validationPath = "/project/s929/pweber/diskValidationData/Re"+case+".txt"
    # validationData = np.loadtxt(validationPath, delimiter=",")
    # plt.plot(validationData[:,0], validationData[:,1], "2k", label="Koumoutsakos and Leonhard (1995)", zorder=10)
    # plt.xlim([0,10])
    # if case == "40":
    #   plt.xlim([0,10])
    # elif case == "550":
    #   plt.xlim([0,7])
    # elif case == "3000" or case == "9500":
    #   plt.xlim([0,6])
    ###################
    ##################################

    speed = 0.2
    radius = 0.1
    for i in range( len(runname) ):
      plotDragTime( rootSCRATCH, runname[i], speed, radius, i, j )

    if case == "40" or case == "200":
      plt.ylim([0,8])
    ## for initial time
    if case == "40":
      plt.ylim([0,25])
    elif case == "1000":
      plt.ylim([0,4])
    elif case == "9500":
      plt.ylim([0,3])
    else: 
      plt.ylim([0.25,2])
    # plt.ylim([0,8])

    plt.xlabel("Time $T=tu_\infty/r$")
    plt.ylabel("Drag Coefficient $C_D=|F_x|/ru_\infty^2$")
    plt.title("Re={}".format(case))
    plt.grid(b=True, which='major', color="white", linestyle='-')
    plt.legend(loc = 'upper right')
    plt.xscale("log")
    plt.yscale("log")
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
  T = 1
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
    tEnd = time[-1]
    indices = (time>T/2) & (time<tEnd-T/2)
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

def plotSwimmerForces():
  # Helper function to rotate
  def rotate( x, y, theta ):
    xRot = x*np.cos(-theta) - y*np.sin(-theta)
    yRot = x*np.sin(-theta) + y*np.cos(-theta)
    return xRot, yRot

  case = ["Re100", "Re158", "Re251", "Re398", "Re631", "Re1000", "Re1585", "Re2512", "Re3981", "Re6310", "Re10000", "Re100000"]
  # case = ["Re100"]
  # levels = "05"
  levels = ["05", "06", "07"]
  L = 0.2
  T = 1
  rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
  rootPROJECT = "/project/s929/pweber/CUP2Damr/mollified-chi/stefanFish/"

  # runname = [ "stefanFish{}_levels".format(Re)+levels for Re in case ]
  runname = [ "stefanFish{}_levels".format("Re1000")+level for level in levels ]

  for i, run in enumerate(runname):
    data   = np.loadtxt(rootPROJECT+run+"/velocity_0.dat", skiprows=1)
    time   = data[:,0]
    angle  = data[:,6]
    forceX = data[:,1]
    forceY = data[:,2]

    # compute average quantities (in [T/2,..,])
    tEnd = time[-1]
    indices = (time>T/2) & (time<tEnd-T/2)
    timeAv  = time[indices]
    # plot rawdata
    plt.plot( timeAv, forceX[indices], color=lighten_color(colors[i],1) , linestyle="--" )
    plt.plot( timeAv, forceY[indices], color=lighten_color(colors[i],0.7), linestyle="--" )
    # forceXAv = []
    # forceYAv = []
    # for t in timeAv:
    #   indices = (time>t-1/2) & (time<t+1/2)
    #   angleAverage = np.mean( angle[indices] )
    #   xAv = np.mean( forceX[indices] )
    #   yAv = np.mean( forceY[indices] )
    #   xAvRot, yAvRot = rotate( xAv, yAv, angleAverage )
    #   forceXAv.append( xAvRot )
    #   forceYAv.append( yAvRot )

    # plt.plot(timeAv, forceXAv, color=lighten_color(colors[i],1) , label="$f_{\parallel}$")
    # plt.plot(timeAv, forceYAv, color=lighten_color(colors[i],0.5), label="$f_{\perp}$")
    # # plt.plot(timeAv, angleAv)

  # plt.plot( time, angle )
  plt.xlabel("Time $t$")
  plt.ylabel("Force $f/(L/T)$")
  plt.grid(b=True, which='major', color="white", linestyle='-')
  plt.tight_layout()
  plt.title(case[i])
  plt.legend()
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
  rootPROJECT = "/project/s929/pweber/CUP2Damr/stefanFish/"

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

def plotBlocksTime():
  cases   = ["10000"] #[ "40", "550", "3000", "9500"]
  root    = "/scratch/snx3000/pweber/CUP2D/"
  speed = 0.2
  radius = 0.1
  levels = 5
  nEff = 16*8*2**(levels-1)
  for case in cases:
    runname = "diskRe{}_levels{}".format(case,levels)
    data = np.loadtxt(root+runname+"/div.txt", skiprows=5)
    time = data[:,0]*speed/radius
    index = time < 0.1
    numGridpoints = data[:,2]*64 
    plt.plot( time, numGridpoints, label="Re={}".format(case) )
    print("average number of gridpoints"+case, np.mean(numGridpoints[index]))
    print("compression factor"+case, nEff*nEff/np.mean(numGridpoints))

  plt.hlines(y=nEff*nEff, xmin=0, xmax=10, color="black", linestyles="dashed", label="$N_{eff}$", zorder=10)
  plt.xlabel("Time $T=tu/r$")
  plt.ylabel("Number of Gridpoints")
  plt.yscale("log")
  plt.legend(facecolor="white", edgecolor="white", ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.3))
  plt.grid(b=True, which='minor', color="white", linestyle='-')
  plt.tight_layout()
  plt.show()

def gridRefiment():
  def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

  cases  = [ "Re9500"]
  levels = ["09", "04", "05", "06", "07", "08" ]
  root    = "/scratch/snx3000/pweber/CUP2D/"
  root    = "/project/s929/pweber/CUP2Damr/disk/"
  speed = 0.2
  radius = 0.1
  for case in cases:
    h = []
    error = []
    for level in levels:
      # runname = "disk"+case+"_levels"+level+"_poissonTol6"
      runname = "diskRe"+case+"_levels"+level+"_cfl0.2"

      # gridData = np.loadtxt(root+runname+"/div.txt", skiprows=5)
      # timeGridpoints = gridData[:,0]*speed/radius
      # numGridpoints = gridData[:,2]*64

      forceData = np.loadtxt(root+runname+"/forceValues_0.dat", skiprows=1)
      time    = forceData[:,0]*speed/radius
      totDrag = forceData[:,1]/(radius*speed*speed)

      # validationPath = "/project/s929/pweber/diskValidationData/"+case+".txt"
      # validationData = np.loadtxt(validationPath, delimiter=",")
      # validationTimes = validationData[:,0]
      # validationDrags = validationData[:,1]

      # valIndx = find_nearest(validationTimes,2)
      # gridIndx = find_nearest(timeGridpoints, validationTimes[valIndx])
      # dataIndx = find_nearest(time, 2)
      dataIndices = (time > 0.5) & (time < 3)
      # print(validationTimes[valIndx], timeGridpoints[gridIndx], time[dataIndx])
      # print(time[dataIndx])

      h.append( 1/np.sqrt(16*8*2**(int(level)-1)) )
      # h.append( 1/np.sqrt(numGridpoints[gridIndx]) )

      if level == "09":
        # target = totDrag[dataIndx]
        target = np.mean(totDrag[dataIndices])
      # error.append( np.abs(totDrag[dataIndx]-target) )
      error.append( np.abs(np.mean(totDrag[dataIndices])-target) )

    h = np.array(h)
    plt.plot( h, error, "o" )
    # plt.plot( h, 10**1*h**(1), label="1st order", linestyle="--" )
    plt.plot( h, 5*10**2*h**(2), label="2nd order", linewidth=1, linestyle="--" )
    plt.plot( h, 10**4*h**(3), label="3rd order", linewidth=1, linestyle="--" )
    # plt.xlim( [5e4,1e6])
    plt.xlabel("Gridspacing")
    plt.ylabel("Error")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend() #facecolor="white", edgecolor="white", ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.3))
    plt.grid(b=True, which='minor', color="white", linestyle='-')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
  # plotDragTimeCylinder()
  # plotDragTimeNaca()
  # plotLiftTimeNaca()
  # plotDragLiftTimeNaca()
  # plotDragAngleNaca()
  # plotSwimmerSpeed( )
  plotSwimmerForces( )
  # plotSwimmerScaling()
  # plotBlocksTime()
  # gridRefiment()
