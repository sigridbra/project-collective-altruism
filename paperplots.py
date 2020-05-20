
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
from scipy import stats
from scipy.interpolate import griddata
import matplotlib.ticker as ticker

### Varying one variable

#multinitfilelist1 = sorted(glob.glob('./data/multiskew-*'))
#multinitfilelist2 = sorted(glob.glob('./data/multiskew0*')) ## really ugly hack because it's quicker than finding out the correct way, and I need to finish up my PhD on time (don't make this into a habbit!)
#
#multinitraw = []
#
#for f in multinitfilelist1 :
#    multinitraw.append(np.loadtxt(f,usecols=[0],delimiter=',',skiprows=1))
#multinitraw.reverse()
#for f in multinitfilelist2 :
#    multinitraw.append(np.loadtxt(f,usecols=[0],delimiter=',',skiprows=1))
#
#multinitavg = np.array(multinitraw)         ## ROW MATRIX
#
#figmultinit, ax = plt.subplots()
#for line in multinitavg : 
#    ax.plot(line,color='#ff7f0e')
#    ax.set(xlabel='timestep',ylabel='$\\langle$ state $\\rangle$')
#    ax.set_ylim([-1,1])
#    
#figmultinit.savefig('multipol.png')

### Standard plot

statesraw = np.loadtxt(open('./data/states.csv',"rb"),delimiter=',',skiprows=1)

statesavg = statesraw[:,0]
statesstd = statesraw[:,1]
clustrstd = statesraw[:,2]

figstates, ax =  plt.subplots()
f1 = ax.plot(statesavg, color='#ff7f0e', label="AVG state")
f2 = ax.plot(statesstd, color='#ff7f0e', alpha=0.5, label="SD states")
f3 = ax.plot(clustrstd, color='#ff7f0e', linestyle=":", label="SD clusters")
#ax.axhline(y=0.0, color='r', linestyle='-')
ax.set(xlabel='timestep',ylabel='AVG // SD')
ax.legend(loc = 'lower right')
ax.xaxis.grid(True, linestyle='dotted')
ax.yaxis.grid(True, linestyle='dotted')
ax.set_ylim([-1,1])

figstates.savefig('states.png')

### Evolving STD

## This was all crap, because data was not normally distributed, still, nice code to keep around, so it'll stay

#stdtime = np.loadtxt('./data/runs.csv',delimiter=',')
#
#times = range (1,np.size(stdtime[0])+1)
#times = np.array(times)
#
#stds = np.apply_along_axis(np.std,axis=0,arr=stdtime)
#
#sigma1 = stds
#sigma2 = stds*2
#
#figstd, ax = plt.subplots() 
#
#ax.fill_between(times,sigma2,-sigma2,zorder=2)
#ax.fill_between(times,sigma1,-sigma1,zorder=2)
#
#figstd.savefig('stdtime1.png')
#
#meanarr = np.apply_along_axis(np.mean,axis=0,arr=stdtime)
#
#figmean, ax = plt.subplots()
#ax.plot(meanarr)
#
#figmean.savefig('mean.png')
#
#figstd, ax = plt.subplots()
#
#for line in stdtime :
#    ax.plot(line,zorder=2)
#
#shiftsigma11 = -stds + meanarr
#shiftsigma12 = stds + meanarr
#shiftsigma21 = -2*stds + meanarr
#shiftsigma22 = 2*stds + meanarr
#
#plt.ylim(-1,1)
#ax.fill_between(times,shiftsigma21,shiftsigma22,zorder=2,color='k',alpha=0.2,label='STD = $1\\sigma$')
#ax.fill_between(times,shiftsigma11,shiftsigma12,zorder=2,color='k',alpha=0.2,label='STD = $2\\sigma$')
#plt.legend()
#
#figstd.savefig('stdtime2.png')

### heatmap

#heatdataraw = np.loadtxt('./data/runs.csv',delimiter=',')
#
#elsinrow = len(heatdataraw[0])
#ts = range (elsinrow)
#times = []
#
#for x in range (len(heatdataraw)) :
#    times.append(ts)
#
#times = np.array(times)
#times = times.flatten()
#heatdata = heatdataraw.flatten()
#
#binsize = 100
#xbins = np.arange(0,elsinrow,binsize)
#ybins = np.linspace(-1,1,elsinrow/binsize)
#
#def isnan (n) :
#    return n != n
#
#heatdatabinned = stats.binned_statistic_2d(times,heatdata,heatdata,'count',bins=[xbins,ybins])
#
#heatdatabinned = np.array(heatdatabinned.statistic)
#heatdatabinned = heatdatabinned.T
#
##for el in np.nditer(heatdata,op_flags=['readwrite']) :
##    if isnan(el) :
##        el[...] = 0
#
#xbins = xbins + (xbins[1]-xbins[0])/2
#ybins = ybins + (ybins[1]-ybins[0])/2
#
#xbins = xbins[:-1]
#ybins = ybins[:-1]
#
#heatmap, ax = plt.subplots()
#
##im = ax.pcolormesh(xbins,ybins,heatdatabinned,cmap='inferno',shading='gouraud',vmin=0,vmax=binsize*10)
#im = ax.pcolormesh(xbins,ybins,heatdatabinned,cmap='inferno',vmin=0,vmax=binsize*10)
#ax.set(xlabel='timestep', ylabel='$\\langle$ state $\\rangle$')
#
#cbar = heatmap.colorbar(im)
#cbar.ax.set_ylabel('counts')
#
#heatmap.savefig('heatmap.png')
#
#### multipol
#
statesraw1 = np.loadtxt(open('./data/stateshigh.csv',"rb"),delimiter=',',skiprows=1)
statesraw2 = np.loadtxt(open('./data/statesmid.csv',"rb"),delimiter=',',skiprows=1)
statesraw3 = np.loadtxt(open('./data/states.csv',"rb"),delimiter=',',skiprows=1)

statesavg1 = statesraw1[:,0]
statesavg2 = statesraw2[:,0]
statesavg3 = statesraw3[:,0]

figstates, ax =  plt.subplots()
f1 = ax.plot(statesavg1, color='#ff7f0e', label="top: $\\phi = 0.027$")
f2 = ax.plot(statesavg2, color='#ff7f0e', label="mid: $\\phi = 0.0135$")
f3 = ax.plot(statesavg3, color='#ff7f0e', label="bot: $\\phi = 0.00675$")
ax.set(xlabel='timestep',ylabel='$\\langle$ state $\\rangle$')
ax.legend(loc = 'lower right')
#ax.yaxis.set_major_locator(ticker.MultipleLocator(0.25))
#ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
#ax.yaxis.set_major_locator(ticker.MultipleLocator(6000))
#ax.xaxis.set_major_locator(ticker.MultipleLocator(2000))
ax.xaxis.grid(True, linestyle='dotted')
ax.yaxis.grid(True, linestyle='dotted')
ax.set_ylim([-1,1])

figstates.savefig('multipol.png')

### multiinit

#figmultinitcolors, ax = plt.subplots()
#
#skews = np.loadtxt(open('./data/skews.dat',"rb"),delimiter=',')
#
#skews = sorted(skews)
#
#acmap = cm.get_cmap('plasma') 
#sm = plt.cm.ScalarMappable(cmap=acmap, norm=plt.Normalize(vmin=np.min(skews), vmax=np.max(skews)))
#
#skews = (skews - np.min(skews)) / (np.max(skews) - np.min(skews))
#
#tupled = tuple(zip(multinitraw,skews))
#
#for (y,z) in tupled :
#    ax.plot(y, color=acmap(z))
#
#ax.set(xlabel='timestep', ylabel='$\\langle$ state $\\rangle$')
#
#cbar = figmultinitcolors.colorbar(sm)
#cbar.ax.set_ylabel('$\phi$')
#
#figmultinitcolors.savefig('multiinitcolor.png')

