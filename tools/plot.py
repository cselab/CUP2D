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

################################## UTILS ##################################
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

def plotSwimmerSpeed():
  # Helper function to rotate
  def rotate( x, y, theta ):
    xRot = x*np.cos(-theta) - y*np.sin(-theta)
    yRot = x*np.sin(-theta) + y*np.cos(-theta)
    return xRot, yRot

  Ts = [0.16, 0.2, 0.25, 0.33, 0.5, 1]
  colorsGreen = []
  colorsBlue = []
  velx = []
  vely = []
  bar = Bar('Processing', max=len(Ts))
  for T in Ts:
    data = np.loadtxt("validation/Swimmer/T={:.02f}/velocity_0.dat".format(T),skiprows=1)
    time = data[:,0]
    phi = data[:,6]
    xvel = data[:,7]
    yvel = data[:,8]
    
    indices = (time>T/2) & (time<11+T/2)
    timeRestricted = time[indices]
    xvelRestricted = xvel[indices]
    yvelRestricted = yvel[indices]
    
    phiAverages = []
    uAverages = []
    vAverages = []
    xVelCorr = []
    yVelCorr = []
    for t, u, v in zip(timeRestricted, xvelRestricted, yvelRestricted):
      indices = (time>t-T/2) & (time<t+T/2)
      phiAverage = np.mean(phi[indices])
      phiAverages.append( phiAverage )
      xRot, yRot = rotate( u, v, phiAverage )
      xVelCorr.append(xRot)
      yVelCorr.append(yRot)
    
    xVelCorr = np.array(xVelCorr)
    yVelCorr = np.array(yVelCorr)
    indicesRestricted = (timeRestricted>T) & (timeRestricted<11)
    timeSuperRestricted = timeRestricted[indicesRestricted]
    for t in timeSuperRestricted:
      indices = (timeRestricted>t-T/2) & (timeRestricted<t+T/2)
      uAverages.append( np.mean( xVelCorr[indices] ) )
      vAverages.append( np.mean( yVelCorr[indices] ) )
   
    uAverages = np.array( uAverages )
    vAverages = np.array( vAverages ) 
    #plt.plot(time, -xvel, color='green', linestyle='dashed')
    #plt.plot(time, yvel, color='blue' , linestyle='dashed')
    #plt.plot(time, phi,  color='red'  , linestyle='dashed')
    # colorGreen = lighten_color("green",T)
    # colorsGreen.append(colorGreen)
    # colorBlue = lighten_color("blue",T)
    # colorsBlue.append(colorBlue)

    #plt.plot(timeRestricted, -xVelCorr, color=colorGreen)
    #plt.plot(timeRestricted, yVelCorr, color=colorBlue)
    #plt.plot(timeRestricted, phiAverages,  color='darkred', label='$\phi$')
    # plt.plot(timeSuperRestricted, -uAverages, color=colorGreen)
    # plt.plot(timeSuperRestricted, vAverages, color=colorBlue)
  
    velx.append(-uAverages[-1])
    # vely.append(vAverages[-1])

    bar.next()
  bar.finish()
  
  # numValues = len(Ts)
  # line = [[(0, 0)]]
  # style = ["solid"]
  # lc1 = mcol.LineCollection(numValues * line, linestyles=numValues * style, colors=colorsGreen)
  # lc2 = mcol.LineCollection(numValues * line, linestyles=numValues * style, colors=colorsBlue)
  # # create the legend
  # plt.legend([lc1, lc2], ['$U_{\parallel}$', '$U_{\perp}$'], handler_map={type(lc1): HandlerDashedLines()},
  #         handlelength=2.5, handleheight=3, ncol=2)
  plt.plot(Ts, velx, "k.")
  plt.xlabel("$T$")
  plt.ylabel("$U_{\parallel}$")
  plt.show()
  #plt.savefig("SwimmerVelocity.png", dpi=300, bbox_inches="tight")

stId = 5

def plotDragLift():
  fig, axs = plt.subplots(1, 3, figsize=(14.5, 5)) #, sharey=True)
  # load data
  tRampup = 0.15
  tCutoffLower = 20
  tCutoffUpper = 40
  startAngle = 0 #8
  endAngle = 7 #14
  thetas = np.arange(startAngle,endAngle+1)
  # create containers
  meanDrag     = []
  stdDrag      = []
  meanLift     = []
  stdLift      = []
  meanQuotient = []
  stdQuotient  = []
  colorsGreen = []
  colorsBlue  = []
  bar = Bar('Processing', max=len(thetas))
  for theta in thetas:
    data = np.loadtxt("validation/NACA/theta/bpdx=96/forceValues_theta={:02d}.dat".format(theta), skiprows=1)
    # plot instantaneous quantities
    colorGreen = lighten_color("green",0.3+theta/5) #-1.2+theta/5
    colorsGreen.append(colorGreen)
    colorBlue = lighten_color("blue",0.3+theta/5)   #-1.2+theta/5
    colorsBlue.append(colorBlue)
    tThresholdIdxRampup = [i for i, x in enumerate(data[:,0] < tRampup) if not x][0]
    tThresholdIdxLower = [i for i, x in enumerate(data[:,0] < tCutoffLower) if not x][0]
    tThresholdIdxUpper = [i for i, x in enumerate(data[:,0] < tCutoffUpper) if not x][0]
    axs[0].plot(data[tThresholdIdxRampup:tThresholdIdxLower,0], data[tThresholdIdxRampup:tThresholdIdxLower,10], color=colorGreen)
    axs[0].plot(data[tThresholdIdxRampup:tThresholdIdxLower,0], data[tThresholdIdxRampup:tThresholdIdxLower,12], color=colorBlue)
    axs[0].ticklabel_format(scilimits=(0,0),axis='y')
    # compute mean quantities
    meanDrag.append(np.mean(data[tThresholdIdxLower:tThresholdIdxUpper,10]))
    stdDrag.append(np.std(data[tThresholdIdxLower:tThresholdIdxUpper,10]))
    meanLift.append(np.mean(data[tThresholdIdxLower:tThresholdIdxUpper,12]))
    stdLift.append(np.std(data[tThresholdIdxLower:tThresholdIdxUpper,12]))
    meanQuotient.append(np.mean(-data[tThresholdIdxLower:tThresholdIdxUpper,12]/data[tThresholdIdxLower:tThresholdIdxUpper,10]))
    stdQuotient.append(np.std(-data[tThresholdIdxLower:tThresholdIdxUpper,12]/data[tThresholdIdxLower:tThresholdIdxUpper,10]))
    bar.next()
  bar.finish()
  
  # plot mean quantities
  axs[1].errorbar(thetas, meanDrag, stdDrag, marker=".", capsize=1.5, label="Drag $F_D$", color="green")
  axs[1].errorbar(thetas, meanLift, stdLift, marker=".", capsize=1.5, label="Lift $F_L$", color="blue")
  axs[1].ticklabel_format(scilimits=(0,0),axis='y')

  axs[2].errorbar(thetas, meanQuotient, stdQuotient, marker=".", capsize=1.5,  color="black")

  # setup axis configuration
  axs[0].set_xlabel("time t [s]")
  axs[0].set_ylabel("${F_D}$ and ${F_L}$")
  # axs[0].set_ylim([-0.0011,0.0004])
  # make list of one line -- doesn't matter what the coordinates are
  line = [[(0, 0)]]
  style = ["solid"]
  # set up the proxy artist
  numValues = endAngle - startAngle + 1
  lc1 = mcol.LineCollection(numValues * line, linestyles=numValues * style, colors=colorsGreen)
  lc2 = mcol.LineCollection(numValues * line, linestyles=numValues * style, colors=colorsBlue)
  # create the legend
  axs[0].legend([lc1, lc2], ['Drag', 'Lift'], handler_map={type(lc1): HandlerDashedLines()},
          handlelength=2.5, handleheight=3, ncol=2)
  
  axs[1].set_xlabel("angle $\\theta$ [deg]")
  axs[1].set_ylabel("$\\overline{F_D}$ and $\\overline{F_L}$")
  axs[1].legend(frameon=False) #bbox_to_anchor=(0.5, -0.1), ncol=2, 
  axs[1].xaxis.set_major_locator(MultipleLocator(5))
  axs[1].xaxis.set_minor_locator(MultipleLocator(1))
  axs[1].set_xlim([0,30])
  
  axs[2].set_xlabel("angle $\\theta$ [deg]")
  axs[2].set_ylabel("$\\overline{F_L/F_D}$")
  axs[2].xaxis.set_major_locator(MultipleLocator(5))
  axs[2].xaxis.set_minor_locator(MultipleLocator(1))
  axs[2].yaxis.set_major_locator(MultipleLocator(1))
  axs[2].yaxis.set_minor_locator(MultipleLocator(0.1))
  axs[2].set_ylim([0,7])
  axs[2].set_xlim([0,30])
  # axs[2].set_yticks(meanQuotient[:7]+[meanQuotient[9]] +[meanQuotient[-2]])
  # axs[2].tick_params(axis='both', which='major', labelsize=6)
  axs[2].grid( b = True, axis = "y" )
  
  # fig.subplots_adjust(wspace=0, hspace=0)

  plt.tight_layout( w_pad = 0 )
  # plt.show()
  plt.savefig( "LiftDrag.eps", transparent=True)

def plotConvergence():
  fig, axs = plt.subplots(1, 3, figsize=(14.5, 5)) #, sharey=True)
  fig.title("$\\theta=14$ deg", fontsize=16)
  gridSizes = [ 24, 32, 48, 64, 96, 128 ]
  numValues = len(gridSizes)
  # load data
  tCutoffLower = 10
  tCutoffUpper = 24
  # create containers
  meanDrag     = []
  stdDrag      = []
  meanLift     = []
  stdLift      = []
  meanQuotient = []
  stdQuotient  = []
  colorsGreen = []
  colorsBlue  = []
  bar = Bar('Processing', max=numValues)
  for i, gridSize in enumerate(gridSizes):
    data = np.loadtxt("validation/resolution/forceValues_bpdx={}.dat".format(gridSize), skiprows=1)
    # plot instantaneous quantities
    colorGreen = lighten_color("green",0.8+i/5)
    colorsGreen.append(colorGreen)
    colorBlue = lighten_color("blue",0.8+i/5)
    colorsBlue.append(colorBlue)
    tThresholdIdxLower = [i for i, x in enumerate(data[:,0] < tCutoffLower) if not x][0]
    tThresholdIdxUpper = [i for i, x in enumerate(data[:,0] < tCutoffUpper) if not x][0]
    axs[0].plot(data[:tThresholdIdxLower,0], data[:tThresholdIdxLower,10], color=colorGreen)
    axs[0].plot(data[:tThresholdIdxLower,0], data[:tThresholdIdxLower,12], color=colorBlue)
    axs[0].ticklabel_format(scilimits=(0,0),axis='y')
    # compute mean quantities
    meanDrag.append(np.mean(data[tThresholdIdxLower:tThresholdIdxUpper,10]))
    stdDrag.append(np.std(data[tThresholdIdxLower:tThresholdIdxUpper,10]))
    meanLift.append(np.mean(data[tThresholdIdxLower:tThresholdIdxUpper,12]))
    stdLift.append(np.std(data[tThresholdIdxLower:tThresholdIdxUpper,12]))
    meanQuotient.append(np.mean(-data[tThresholdIdxLower:tThresholdIdxUpper,12]/data[tThresholdIdxLower:tThresholdIdxUpper,10]))
    stdQuotient.append(np.std(-data[tThresholdIdxLower:tThresholdIdxUpper,12]/data[tThresholdIdxLower:tThresholdIdxUpper,10]))
    bar.next()
  bar.finish()
  
  # plot mean quantities
  axs[1].errorbar(gridSizes, meanDrag, stdDrag, marker=".", capsize=1.5, label="Drag C_D", color="green")
  axs[1].errorbar(gridSizes, meanLift, stdLift, marker=".", capsize=1.5, label="Lift C_L", color="blue")
  axs[1].ticklabel_format(scilimits=(0,0),axis='y')

  axs[2].errorbar(gridSizes, meanQuotient, stdQuotient, marker=".", capsize=1.5,  color="black")

  # set axis labels
  axs[0].set_xlabel("time t [s]")
  axs[0].set_ylabel("${C_D}$ and ${C_L}$")
  # make list of one line -- doesn't matter what the coordinates are
  line = [[(0, 0)]]
  style = ["solid"]
  # set up the proxy artist
  lc1 = mcol.LineCollection(numValues * line, linestyles=numValues * style, colors=colorsGreen)
  lc2 = mcol.LineCollection(numValues * line, linestyles=numValues * style, colors=colorsBlue)
  # create the legend
  axs[0].legend([lc1, lc2], ['Drag', 'Lift'], handler_map={type(lc1): HandlerDashedLines()},
          handlelength=2.5, handleheight=3,frameon=False)
  
  axs[1].set_xlabel("bpdx")
  axs[1].set_ylabel("$\\overline{C_D}$ and $\\overline{C_L}$")
  axs[1].legend(frameon=False) #bbox_to_anchor=(0.5, -0.1), ncol=2, 
  axs[1].set_xticks(gridSizes, rotation=45)

  axs[2].set_xlabel("bpdx")
  axs[2].set_ylabel("$\\overline{C_L/C_D}$")
  axs[2].set_xticks(gridSizes, rotation=45)
  
  # fig.subplots_adjust(wspace=0, hspace=0)

  plt.tight_layout( ) #w_pad = 0.1 )
  # plt.show()
  plt.savefig( "GridConvergence.eps", transparent=True)

def plotDragTime( root, runname, speed, radius ):
  data = np.loadtxt(root+runname+"/forceValues_0.dat", skiprows=1)
  time = data[:,0]
  drag = data[:,10]
  drag = drag/(radius*speed*speed)
  plt.plot(time, drag, label=runname+", $C_D(t=10)=${}".format(drag[-1]))
  plt.ylim([1.5,2.5])
  # plt.xlim([0,10])
  plt.xlabel("time $t$")
  plt.ylabel("Drag Coefficient $C_D=F/ru_\infty^2$")
  plt.grid()
  # plt.title("$C_D(t=10)=${}".format(drag[-1]))
  


if __name__ == '__main__':
  # plotSwimmerSpeed( )
  # plotCoM()
  # plotDragLift()
  # plotConvergence()
  rootSCRATCH = "/scratch/snx3000/pweber/CUP2D/"
  rootPROJECT = "/project/s929/pweber/"
  # runname = ["diskRe40_32x16_levelMax=1", "diskRe40_48x24_levelMax=1", "diskRe40_64x32_levelMax=1", "diskRe40_96x48_levelMax=1", "diskRe40_128x64_levelMax=1", "diskRe40_8x4_levelMax=3", "diskRe40_8x4_levelMax=4", "diskRe40_8x4_levelMax=5", "diskRe40_6x3_levelMax=4", "diskRe40_6x3_levelMax=5"]
  runname = ["diskRe40_128x64_levelMax=1", "diskRe40_8x4_levelMax=5"]
  speed = 0.15
  radius = 0.0375
  for i in range( len(runname) ):
    plotDragTime( rootSCRATCH, runname[i], speed, radius )
  plt.legend()
  plt.show()
