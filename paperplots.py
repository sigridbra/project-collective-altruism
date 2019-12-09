
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
from scipy import stats
from scipy.interpolate import griddata

### Varying one variable

#multinitfilelist = glob.glob('./data/skew*')
#multinitraw = []
#
#for f in multinitfilelist :
#    multinitraw.append(np.loadtxt(f,usecols=[0],delimiter=',',skiprows=1))
#
#multinitavg = np.array(multinitraw)         ## ROW MATRIX
#
#figmultinit, ax = plt.subplots()
#for line in multinitavg : 
#    ax.plot(line)
#
#figmultinit.savefig('multiinit.png')

### Standard plot

#statesraw = np.loadtxt(open('./data/states.csv',"rb"),delimiter=',',skiprows=1)

#statesavg = statesraw[:,0]
#statesstd = statesraw[:,1]
#clustrstd = statesraw[:,2]
#
#figstates, ax =  plt.subplots()
#ax.plot(statesavg)
#
#figstates.savefig('states.png')

### Evolving STD

# This was all crap, because data was not normally distributed, still, nice code to keep around, so it'll stay

stdtime = np.loadtxt('./data/runs.csv',delimiter=',')

times = range (1,np.size(stdtime[0])+1)
times = np.array(times)

stds = np.apply_along_axis(np.std,axis=0,arr=stdtime)

sigma1 = stds
sigma2 = stds*2

figstd, ax = plt.subplots() 

ax.fill_between(times,sigma2,-sigma2,zorder=2)
ax.fill_between(times,sigma1,-sigma1,zorder=2)

figstd.savefig('stdtime1.png')

meanarr = np.apply_along_axis(np.mean,axis=0,arr=stdtime)

figmean, ax = plt.subplots()
ax.plot(meanarr)

figmean.savefig('mean.png')

figstd, ax = plt.subplots()

for line in stdtime :
    ax.plot(line,zorder=2)

#shiftsigma11 = -stds + meanarr
#shiftsigma12 = stds + meanarr
#shiftsigma21 = -2*stds + meanarr
#shiftsigma22 = 2*stds + meanarr
#
#plt.ylim(-1,1)
#ax.fill_between(times,shiftsigma21,shiftsigma22,zorder=2,color='k',alpha=0.2,label='STD = $1\\sigma$')
#ax.fill_between(times,shiftsigma11,shiftsigma12,zorder=2,color='k',alpha=0.2,label='STD = $2\\sigma$')
#plt.legend()

figstd.savefig('stdtime2.png')

### heatmap

heatdataraw = np.loadtxt('./data/runs.csv',delimiter=',')

elsinrow = len(heatdataraw[0])
ts = range (elsinrow)
times = []

for x in range (len(heatdataraw)) :
    times.append(ts)

times = np.array(times)
times = times.flatten()
heatdata = heatdataraw.flatten()

binsize = 100
xbins = np.arange(0,elsinrow,binsize)
ybins = np.linspace(-1,1,elsinrow/binsize)

def isnan (n) :
    return n != n

heatdatabinned = stats.binned_statistic_2d(times,heatdata,heatdata,'count',bins=[xbins,ybins])

heatdatabinned = np.array(heatdatabinned.statistic)
heatdatabinned = heatdatabinned.T

#for el in np.nditer(heatdata,op_flags=['readwrite']) :
#    if isnan(el) :
#        el[...] = 0

xbins = xbins + (xbins[1]-xbins[0])/2
ybins = ybins + (ybins[1]-ybins[0])/2

xbins = xbins[:-1]
ybins = ybins[:-1]

heatmap, ax = plt.subplots()

#im = ax.pcolormesh(xbins,ybins,heatdatabinned,cmap='inferno',shading='gouraud',vmin=0,vmax=binsize*10)
im = ax.pcolormesh(xbins,ybins,heatdatabinned,cmap='inferno',vmin=0,vmax=binsize*10)

heatmap.colorbar(im)

heatmap.savefig('heatmap.png')

