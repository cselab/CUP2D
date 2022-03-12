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
sns.set_style("whitegrid")

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
###########################################################################

def plotForceTime( axs, root, runname, radius, i, j ):
  ## load kinematic data ##
  kineticData = np.loadtxt(root+runname+"/velocity_0.dat", skiprows=1)
  time = kineticData[:,0]
  u = kineticData[:,7]
  v = kineticData[:,8]
  omega = kineticData[:,9]
  # print("speed:", u, v)
  #########################

  ## load dynamic data ##
  dynamicData = np.loadtxt(root+runname+"/forceValues_0.dat", skiprows=1)
  dynamicTime = dynamicData[:,0]
  # np.testing.assert_array_equal(dynamicTime, time)
  forceX  = dynamicData[:,1]
  forceY  = dynamicData[:,2]
  tau     = dynamicData[:,7]
  # print("force:", forceX, forceY)
  #######################

  ## load power data ##
  powerData = np.loadtxt(root+runname+"/powerValues_0.dat", skiprows=1)
  powerTime    = powerData[:,0]
  # np.testing.assert_array_equal(powerTime, time)
  pOut  = powerData[:,5]
  for iii in range (len(u)):
      pOut[iii] = (u[iii]*forceX[iii]+v[iii]*forceY[iii]) + tau[iii]*omega[iii]
  # print("force:", forceX, forceY)
  #######################

  ## compute object speed ##
  # speed = np.sqrt(u**2 + v**2)
  speed = 0.2
  ##########################

  ## dimensionless time ##
  time    = time #*speed/radius
  ########################

  ## compute dimensionless drag ##
  # totDrag = (u*forceX+v*forceY) / speed
  totDrag = forceX
  dragCoeff = np.abs(totDrag) / (radius*speed*speed)
  # print("drag:", totDrag, dragCoeff)
  ##################

  ## compute dimensionless lift ##
  # totLift = (-v*forceX+u*forceY) / speed
  totLift = forceY
  liftCoeff = -totLift / (radius*speed*speed)
  # print("lift:", totLift, liftCoeff)
  ##################

  ## compute dimensionless power ##
  powerCoeff = -pOut / (radius*speed*speed*speed)
  # print("lift:", totLift, liftCoeff)
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

  
  # print(time[0], dragCoeff[0])
  # if i == 2: 
  #   index = dragCoeff < 2.0
  #   print(time[index])
  #   plt.plot(time[index], dragCoeff[index],  color=lighten_color(colors[i],1), label="$C_D$ present ({} levels)".format(i+4) )# , label=runname+", $C_D")
  #################################################################################
  #### uncomment and adapt i to levelMax for which you want to plot the result ####
  # if i == j+2:
  minLevels = 6
  ## plot drag ##
  if j == 0:
    axs.plot(dynamicTime[::10], dragCoeff[::10], color=lighten_color(colors[i],1), label="{} levels".format(i+minLevels))
    # indices = (dynamicTime > 0.5) & (dynamicTime < 4.25)
    # print('[CUP2D] Average Drag $t\\in [0.50,4.25]$: ', sp.integrate.simps(dragCoeff[indices], dynamicTime[indices]) / (4.25-0.5) )
  ########################

  ## plot histogram of drag ##
  if j == 1:
    index = dynamicTime > 2
    axs = sns.distplot(dragCoeff[index])

  ## plot lift ##
  # if j == 1:
  #   axs[j].plot(dynamicTime, liftCoeff, color=lighten_color(colors[i],1), label="present ({} levels)".format(i+minLevels))
  #   indices = (dynamicTime > 0.68) & (dynamicTime < 4.57)
  #   print("[CUP2D] Average Lift $t\\in [0.68,4.57]$: ", sp.integrate.simps(liftCoeff[indices], dynamicTime[indices]) / (4.57-0.68) )
  ########################

  ## plot power ##
  # if j == 2:
  #   axs[j].plot(powerTime, powerCoeff, color=lighten_color(colors[i],1), label="present ({} levels)".format(i+minLevels))
  #   indices = (dynamicTime > 0.36) & (dynamicTime < 4.14)
  #   print("[CUP2D] Average Power $t\\in [0.36,4.14]$: ", sp.integrate.simps(powerCoeff[indices], dynamicTime[indices]) / (4.14-0.36) )
  ########################
  
  ## plot autocorrelation of drag/lift to detect frequency ##
  # autoCorrDragCoeff = np.correlate(dragCoeff, dragCoeff, mode='full')
  # autoCorrLiftCoeff = np.correlate(liftCoeff, liftCoeff, mode='full')
  # # print(autoCorrDragCoeff[autoCorrDragCoeff.size//2:])
  # # print(autoCorrLiftCoeff[autoCorrLiftCoeff.size//2:])
  # print("angle=",i)
  # indices = time>50
  # plt.plot(time[indices], autoCorrDragCoeff[autoCorrDragCoeff.size//2:][indices])
  # plt.plot(time[indices], autoCorrLiftCoeff[autoCorrLiftCoeff.size//2:][indices])
  ###########################################################

  ## for naca plot drag and lift ##
  # plt.plot(time, dragCoeff,  color=lighten_color("green",0.25+i/4))
  # plt.plot(time, liftCoeff,  color=lighten_color("blue",0.25+i/4) )
  #################################
  
def plotDragTimeCylinder():
  # "The initial flow past an impulsively started circular cylinder", W. M. Collins and S. C. R. Dennis (1973)
  def dragCollinsDennis( Re, t ):
    k = 2*np.sqrt(2*t/Re)
    fricDrag = np.pi*(1/np.sqrt(2*Re*t))*(2.257+k-0.141*k**2+0.062*k**3 - (0.092-1.6*k+6.9*k**2-22.7*k**3)*t**2 - (0.074-1.24*k+12.12*k**2+24.35*k**3)*t**4 + (0.008+0.196*k)*t**6 )
    presDrag = np.pi*(1/np.sqrt(2*Re*t))*(2.257+k-0.141*k**2+(4.59-22.1*k+78.8*k**2)*t**2 + (2.68-40.5*k+219.3*k**3)*t**4 + 0.894*t**6 )
    # print("diff:", presDrag-fricDrag)
    ### equation (80) + (81) ###
    return fricDrag+presDrag
    ### equation (82) ###
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
  # cases = [ "100000" ]
  ######################

  ## for long time ## "40", "3000", "9500"  
  cases = [ "9500"]
  ######################

  for j, case in enumerate(cases):
    rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
    rootPROJECT = "/project/s929/pweber/CUP2Damr/mollified-chi/disk/"

    runname = [ "diskRe{}_levels{:01d}_dt1e-4".format(case, level) for level in np.arange(4,9) ]
    runname = [ "diskRe{}_levels6_dt{:.0e}".format(case, timestep) for timestep in [ 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4, 2e-4 ] ]

    ###### plot validation data ######
    ##################################

    ######################
    ## for initial time ##
    # time = np.linspace(0,0.5,1001)
    # plt.plot( time, dragCollinsDennis( int(case), time ), linestyle="--", color="black", label="Collins and Dennis (1973)")
    # plt.plot( time, dragBarLevYang( int(case), time ), linestyle="--", color="lightgrey", label="Bar-Lev and Yang (1975)")
    # validationPath = "/project/s929/pweber/diskValidationData/Re"+case+"-start.txt"
    # validationData = np.loadtxt(validationPath, delimiter=",")
    # plt.plot(validationData[:,0], validationData[:,1], "2k", label="Koumoutsakos and Leonhard (1995)")
    plt.xlim([0,0.5])
    ######################

    ###################
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

    radius = 0.1
    for i in range( len(runname) ):
      plotForceTime( rootSCRATCH, runname[i], radius, i, j )

    plt.xlabel("Time $T=tu_\infty/c$")
    plt.ylabel("Thrust Coefficient $C_T=|F_x|/cu_\infty^2$")
    # plt.title("Re={}".format(case))
    plt.grid(b=True, which='major', color="white", linestyle='-')
    plt.legend(loc = 'upper right')
    # plt.xscale("log")
    # plt.yscale("log")
    plt.show()

def plotDragLiftTimeNaca():
  # levels = np.arange(5,8)
  level = 5
  angle  = np.arange(0,31)

  rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
  rootPROJECT = "/project/s929/pweber/CUP2Damr/mollified-chi/naca_highTolAMR/"

  runname = [ "nacaRe1000_levels{:01d}_angle{:02d}".format(level,a) for a in angle ]

  charlength = 0.2/2
  for i in range( len(runname) ):
    plotForceTime( rootSCRATCH, runname[i], charlength, i, 0 )

  plt.ylim([-2,1])
  plt.xlim([0,50])
  plt.xlabel("Time $T=2t u_\infty/c$")
  plt.ylabel("Force Coefficient $C_{D/L}=2F_{x/y}/cu_\infty^2$")
  plt.grid(b=True, which='major', color="white", linestyle='-')
  plt.legend()
  plt.show()

def plotForceAngleNaca():
  levels = [ "8" ] #"5", "7", "8", "9"
  angles = np.arange(15)

  validationQuotient = "/project/s929/pweber/nacaValidationData/Re1000-quotient.txt"
  validationDrag = "/project/s929/pweber/nacaValidationData/Re1000-drag.txt"
  validationPresDrag = "/project/s929/pweber/nacaValidationData/Re1000-pressureDrag.txt"
  validationViscDrag = "/project/s929/pweber/nacaValidationData/Re1000-viscousDrag.txt"
  validationLift = "/project/s929/pweber/nacaValidationData/Re1000-lift.txt"
  validDataQuotient = np.loadtxt(validationQuotient, delimiter=",")
  validDataDrag = np.loadtxt(validationDrag, delimiter=",")
  validDataPresDrag = np.loadtxt(validationPresDrag, delimiter=",")
  validDataViscDrag = np.loadtxt(validationViscDrag, delimiter=",")
  validDataLift = np.loadtxt(validationLift, delimiter=",")

  rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
  rootPROJECT = "/project/s929/pweber/CUP2Damr/naca/"

  # runname = [ "nacaRe1000_levels"+levels+"_angle{:01d}".format(a) for a in case ]

  speed = 0.2
  chordlength = 0.2

  for level in levels:
    avQuotient = []
    avDrag = []
    avLift = []
    for i, angle in enumerate(angles):
      run = "nacaRe1000-usemap" #"nacaRe1000_levels"+level+"_angle{:02d}".format(angle)
      data = np.loadtxt(rootSCRATCH+run+"/forceValues_0.dat", skiprows=1)
      time = data[:,0]*speed/chordlength
      drag = data[:,1]/(chordlength/2*speed*speed)
      presDrag = data[:,3]/(chordlength/2*speed*speed)
      viscDrag = data[:,5]/(chordlength/2*speed*speed)
      print("drag:", presDrag, viscDrag, drag )
      lift = data[:,2]/(chordlength/2*speed*speed)

      ## plot raw-data with averaged values ##
      plt.plot(time, drag,     label="$C_D$")
      plt.plot(time, presDrag, label="$C_P$")
      plt.plot(time, viscDrag, label="$C_V$")
      timeValid = np.linspace( 0, 200, 1001 )
      Cd   = np.full(timeValid.shape, validDataDrag[i,1])
      CdP  = np.full(timeValid.shape, validDataPresDrag[i,1])
      CdV  = np.full(timeValid.shape, validDataViscDrag[i,1])
      plt.plot(timeValid, Cd,  label="$C_D$ Kurtulus (2016)")
      plt.plot(timeValid, CdP, label="$C_P$ Kurtulus (2016)")
      plt.plot(timeValid, CdV, label="$C_V$ Kurtulus (2016)")

      # plt.plot(time, lift, color="green", label="lift present ({} levels)".format(level))
      # Cl   = np.full(timeValid.shape, -validDataLift[i,1])
      # plt.plot(timeValid, Cl, color="black", linestyle="--")
      plt.ylim([0,0.8])
      plt.xlim([0,20])
      plt.legend()
      plt.show()
      ###################

      ## averaging ##
      # averagingIndices = time>(time[-1]-10)
      # avQuotient.append( np.mean(lift[averagingIndices]) / np.mean(drag[averagingIndices]) )
      # avDrag.append( np.mean(drag[averagingIndices]) )
      # avLift.append( np.mean(lift[averagingIndices]) )
      ###############

    ## plot averages ##
    # plt.plot( angles, avQuotient, label="present ({} levels)".format(level))
    # plt.plot( angles, avDrag, label="present ({} levels)".format(level))
    # plt.plot( angles, avLift, label="present ({} levels)".format(level))
    ###################

    # plt.plot( validDataQuotient[:15,0], validDataQuotient[:15,1], "2k", label="Kurtulus (2016)")
    # plt.plot( validDataDrag[:15,0], validDataDrag[:15,1], "2k", label="Kurtulus (2016)")
    # plt.plot( validDataLift[:15,0], validDataLift[:15,1], "2k", label="Kurtulus (2016)")
  
  plt.xlabel("Angle $\\alpha$")
  # plt.ylabel("Drag Coefficient $C_D=2|F_x|/cu_\infty^2$")
  # plt.ylabel("Lift Coefficient $C_L=2|F_y|/cu_\infty^2$")
  plt.ylabel("Force Coefficient $C_L/C_D$")
  plt.grid(b=True, which='major', color="white", linestyle='-')
  plt.tight_layout()
  plt.legend()
  plt.show()

def plotSwimmerSpeed():
  # Helper function to rotate
  def rotate( x, y, theta ):
    xRot = x*np.cos(-theta) - y*np.sin(-theta)
    yRot = x*np.sin(-theta) + y*np.cos(-theta)
    return xRot, yRot

  validationPath = "/project/s929/pweber/fishValidationData/"
  validationParallelData      = np.loadtxt(validationPath+"Kern-2D-parallel.txt", delimiter=",")
  validationPerpendicularData = np.loadtxt(validationPath+"Kern-2D-perpendicular.txt", delimiter=",")

  # case = ["Re100", "Re158", "Re251", "Re398", "Re631", "Re1000", "Re1585", "Re2512", "Re3981", "Re6310", "Re10000", "Re100000"]
  # case = ["Re100"]
  levels = [ "7" ] # "5", "6", "8", "9"
  # levels = "05"
  L = 0.2
  T = 1
  rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
  rootPROJECT = "/project/s929/pweber/CUP2Damr/naca/"

  runname = [ "carlingFish_levels"+level for level in levels ]

  for i, run in enumerate(runname):
    data = np.loadtxt(rootSCRATCH+run+"/velocity_0.dat", skiprows=1)
    time = data[:,0] / T
    angle= data[:,6]
    u    = data[:,7] * T / L
    v    = data[:,8] * T / L

    ## rotate data ##
    u, v = rotate(u, v, angle) 
    #################

    timeCutoff = 0
    # plot data
    plt.plot( time, -u, color=lighten_color(colors[i],1)  , label="present ({} levels)".format(levels[i]) )
    # plt.plot( time, v, color=lighten_color(colors[i],0.7))

    ## to plot averages ##
    # compute average quantities (in [T/2,..,])
    # tEnd = time[-1]
    # indices = (time>T/2) & (time<tEnd-T/2)
    # timeAv  = time[indices]
    # uAv     = []
    # vAv     = []
    # angleAv = []
    # for t in timeAv:
    #   indices = (time>t-1/2) & (time<t+1/2)
    #   angleAverage = np.mean( angle[indices] )
    #   uAverage = np.mean( u[indices] )
    #   vAverage = np.mean( v[indices] )
    #   uAvRot, vAvRot = rotate( uAverage, vAverage, angleAverage )
    #   angleAv.append( angleAverage )
    #   uAv.append( uAvRot )
    #   vAv.append( vAvRot )
    # plt.plot(timeAv, uAv, color="blue" , label="$u_{\parallel}$")
    # plt.plot(timeAv, vAv, color="green", label="$u_{\perp}$")
    # plt.plot(timeAv, angleAv)
    #######################

  # plot validation data
  index = validationParallelData[:,0] < 6
  plt.plot(      validationParallelData[index,0],      validationParallelData[index,1], "2k", label="Kern und Koumoutsakos (2006)")
  # timeValid = np.linspace( 0, 7, 1001 )
  # Cd   = np.full(timeValid.shape, 0.54)
  # plt.plot(timeValid, Cd, "k--")
  # plt.plot( validationPerpendicularData[:,0], validationPerpendicularData[:,1], "2k")

  # plt.plot( time, angle )
  plt.xlabel("Time $t/T$")
  plt.ylabel("Forward Velocity $u_{\parallel}/(L/T)$")
  plt.grid(b=True, which='major', color="white", linestyle='-')
  plt.tight_layout()
  plt.legend()
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

import scipy as sp

def plotBlocksTime():
  cases   = [ "10^5" ]
  root    = "/scratch/snx3000/mchatzim/highRe2D/"
  speed = 0.2
  radius = 0.1
  levels = np.arange(7,11)
  for case in cases:
    runnames = [ "Re{}-{:02d}".format(case, level) for level in levels]
    for i, runname in enumerate(runnames):
      nEff = 16*8*2**(levels[i]-1)*8*8*2**(levels[i]-1)
      data = np.loadtxt(root+runname+"/div.txt", skiprows=levels[i])
      time = data[:,0]*speed/radius
      numGridpoints = data[:,2]*64 
      plt.plot( time, numGridpoints, label="Re={}".format(case) )
      print("compression factor "+runname, nEff/( sp.integrate.simps( numGridpoints, time) / (time[-1]-time[0]) ) )

    plt.hlines(y=nEff, xmin=0, xmax=10, color="black", linestyles="dashed", label="$N_{eff}$", zorder=10)
    plt.xlabel("Time $T=tu/r$")
    plt.ylabel("Number of Gridpoints")
    plt.yscale("log")
    plt.legend(facecolor="white", edgecolor="white", ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.3))
    plt.grid(b=True, which='minor', color="white", linestyle='-')
    plt.tight_layout()
    plt.show()

def gridRefiment():
  cases  = [ "9500"]
  levels = np.arange(4,9)
  levels = levels[::-1]
  timesteps = [ 1e-5, 2e-5, 4e-5, 6e-5, 8e-5, 1e-4, 2e-4 ]
  root    = "/scratch/snx3000/pweber/CUP2D/"
  speed = 0.2
  radius = 0.1
  refined = "time"

  for case in cases:
    h = []
    error = []
    if refined == "space":
      for level in levels:
        runname = "diskRe{}_levels{}_dt1e-4".format(case, level)

        gridData = np.loadtxt(root+runname+"/div.txt", skiprows=level)
        timeGridpoints = gridData[:,0]*speed/radius
        numGridpoints = gridData[:,2]*64

        forceData = np.loadtxt(root+runname+"/forceValues_0.dat", skiprows=1)
        time    = forceData[:,0]*speed/radius
        totDrag = forceData[:,1]/(radius*speed*speed)

        timeStart = 0
        timeEnd = 1
        timeDiff = timeEnd - timeStart
        indicesGrid = (timeGridpoints >= timeStart) & (timeGridpoints <= timeEnd)
        indices = (time >= timeStart) & (time <= timeEnd)

        if level == 8:
          dragTarget = totDrag[indices]
        else:
          errorDrag = np.abs( (totDrag[indices] - dragTarget ) )
          # compute average number of gridpoints
          # h.append( sp.integrate.simps( numGridpoints[indicesGrid], timeGridpoints[indicesGrid] ) / timeDiff )
          h.append( 16*8**2*2**(level-1) )
          #compute mean error drag
          error.append( sp.integrate.simps( errorDrag, time[indices]) / timeDiff )

      h = np.array(h)
      plt.plot( h, error, "o" )
      plt.plot( h, 10**1*h**(-1/2), label="1st order", linestyle="--" )
      plt.plot( h, 3*10**3*h**(-2/2), label="2nd order", linewidth=1, linestyle="--" )
      plt.plot( h, 9*10**5*h**(-3/2), label="3rd order", linewidth=1, linestyle="--" )
      # plt.xlim( [1.3e4,1e5])
      plt.xlabel("Number of Gridpoints")
      plt.ylabel("Error")
      plt.xscale("log", base=2)
      plt.yscale("log")
      plt.legend() #facecolor="white", edgecolor="white", ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.3))
      plt.grid(b=True, which='minor', color="white", linestyle='-')
      plt.tight_layout()
      plt.show()
    elif refined == "time":
      for timestep in timesteps:
        runname = "diskRe{}_levels6_dt{:.0e}".format(case, timestep)

        forceData = np.loadtxt(root+runname+"/forceValues_0.dat", skiprows=1)
        time    = forceData[:,0]*speed/radius
        totDrag = forceData[:,1]/(radius*speed*speed)

        timeStart = 0
        timeEnd = 1
        timeDiff = timeEnd - timeStart
        indices = (time >= timeStart) & (time < timeEnd)
        time = time[indices]

        if timestep == 1e-5:
          dragTarget = totDrag[indices]
          fDragTarget = sp.interpolate.interp1d(time, dragTarget)
        else:
          print(np.abs( totDrag[indices]  - fDragTarget(time) ), sp.integrate.simps( np.abs( totDrag[indices]  - fDragTarget(time) ), time ))
          error.append( sp.integrate.simps( np.abs( totDrag[indices]  - fDragTarget(time) ), time ) )

      h = np.array( timesteps[1:] )
      print((h), (error))
      plt.plot( h, error, "o")
      plt.plot( h, 2*10**0*h, label="1st order", linewidth=1, linestyle="--" )
      plt.plot( h, 1.5*10**5*h**(2), label="2nd order", linewidth=1, linestyle="--" )
      # plt.ylim([0.02,5])
      plt.xscale("log")
      plt.yscale("log")
      plt.xlabel("Timestep")
      plt.ylabel("Error")
      plt.legend() 
      plt.show()

import os

def scaling():
  cores = np.arange(1,13)
  times = []
  sequentialFraction = 0.11
  for i in cores:
    directory = "/cluster/scratch/webepasc/CUP2D/diskRe3000_threads{:03d}/".format(i)
    lsfFiles = [f for f in os.listdir(directory) if f.startswith('lsf')]
    text = open(directory+lsfFiles[0], 'r').read()
    for line in text.split("\n"):
      if 'Runtime' in line:
        time = line.split(" ")[-1]
        times.append(float(time))
  times = np.array(times)
  print(times)
  plt.plot(cores, 1/(sequentialFraction+(1-sequentialFraction)/cores))
  plt.plot(cores, times[0]/times)
  plt.plot(cores, cores, "k--")
  plt.savefig("scaling.png")

def plotForceTimeTeardrop():

  rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"

  #runname = [ "teardropFixedMPI_PR=100_PI=10000_levels{:01d}".format(level) for level in np.arange(7,12) ]
  runname = ["singleNodeRuns/teardropFixed_levels7"]

  ###### plot validation data ######
  ##################################

  validationPath = "/project/s929/pweber/hydrofoilValidationData/fStar-664_Re5400.csv"
  validationData = np.loadtxt(validationPath, delimiter=",", skiprows=1)
  timestep = 1.5060240964 / 500
  time = timestep*validationData[:,0]

  fig, axs = plt.subplots(3, sharex=True)
  for j in range(3):
    if j == 0:
      axs[j].plot(time, -validationData[:,4], "2k", label="reference", zorder=10)
      indices = (time > 0.5) & (time < 4.25)
      print('[Reference] Average Drag $t\\in[0.50,4.25]$: ', sp.integrate.simps( -validationData[indices,4], time[indices]) / (4.25-0.5) )
    if j == 1:
      axs[j].plot(time, -validationData[:,1], "2k", label="reference", zorder=10)
      indices = (time > 0.66) & (time < 4.5)
      print('[Reference] Average Lift $t\\in[0.66,4.50]$: ', sp.integrate.simps( -validationData[indices,1], time[indices]) / (4.5-0.66) ) 
    if j == 2:
      axs[j].plot(time, -validationData[:,3], "2k", label="reference", zorder=10)
      indices = (time > 0.68) & (time < 3.7)
      print('[Reference] Average Power $t\\in [0.68,3.70]$: ', sp.integrate.simps( -validationData[indices,3], time[indices]) / (3.7-0.68) )

    ###### plot simulation data ######
    ##################################

    chordlength = 0.1

    for i in range( len(runname) ):
      plotForceTime( axs, rootSCRATCH, runname[i], chordlength, i, j )

    if j == 0:
      axs[j].set_ylabel("Thrust Coefficient $C_T=2|F_x|/cu_\infty^2$")
      axs[j].set_ylim([-0.25,0.25])
    if j == 1:
      axs[j].set_ylabel("Lift Coefficient $C_L=2|F_y|/cu_\infty^2$")
      axs[j].set_ylim([-4,4])
    if j == 2:
      axs[j].set_ylabel("Power Coefficient $C_P=2|P|/cu_\infty^3$")
      axs[j].set_ylim([-0.1,1])

    axs[j].grid(b=True, which='major', linestyle='-')
    axs[j].set_xlim([0,5])
    # plt.xscale("log")
    # plt.yscale("log")
  axs[2].set_xlabel("Time $t$")
  axs[0].legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=3)
  plt.show()

def plotForceTimeSquare():

  rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"

  # runname = [ "rectangle_lowPoissonTol_levels{:d}".format(level) for level in np.arange(6,10) ]
  # runname = [ "rectangle_e3`lowTol_levels6"]
  # runname = ["singleNodeRuns/teardropFixed_levels7"]

  runname = [ "disk_levels6"]


  fig, ax = plt.subplots(1)

  ###### plot simulation data ######
  ##################################

  chordlength = 0.1

  j = 0
  for i in range( len(runname) ):
      plotForceTime( ax, rootSCRATCH, runname[i], chordlength, i, j )

      if j == 0:
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Drag Coefficient $C_T=2|F_x|/Lu_\infty^2$")
        # axs[j].set_xlim([0,5])
        ax.set_ylim([0,10])

        ax.grid(b=True, which='major', linestyle='-')

      if j == 1:
        ax.set_xlabel("$C_D$")
        ax.set_ylabel("Density")
        ax.set_yscale('log')
        # plt.xscale("log")
        # plt.yscale("log")

      ax.legend()

  plt.show()

if __name__ == '__main__':
  # plotDragTimeCylinder()
  
  # plotForceTimeTeardrop()

  plotForceTimeSquare()
  
  # fig, ax = plt.subplots(1)
  # normalDistribution = np.random.normal(0, 1, 5000)
  # ax = sns.distplot(normalDistribution)
  # ax.set_yscale('log')
  # plt.show()

  # plotDragLiftTimeNaca()
  # plotForceAngleNaca()
  
  # plotSwimmerSpeed( )
  # plotSwimmerForces( )
  # plotSwimmerScaling()

  # scaling()
  
  # plotBlocksTime()
  # gridRefiment()
