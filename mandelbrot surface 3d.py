# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Mandelbrot surface 3d

'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''
# %autosave 0
# %matplotlib widget
import ipywidgets as widgets
from ipywidgets import VBox, Box
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


def mandelbrot(x, y, mI):
    c=0
    z=np.complex(real=x,imag=y)
    c=z
    for i in range(mI):
        if z.real**2+z.imag**2 >= 4:
            break
        else:
            z = z*z + c
    return i, z


mI=15
n=30
itNum = np.zeros((n,n))
Z     = np.zeros((n,n),dtype=np.complex_)
# Make data.
x = np.linspace(-2.25, +0.75, n) # 3
y = np.linspace(-1.5, +1.5, n)
X,Y = np.meshgrid(x, y, sparse=True)
for i,vx in enumerate(x):
    for j,vy in enumerate(y):
        itNum[i][j], Z[i][j] = mandelbrot( vx, vy, mI)


# # replicate

# +
def make_box_layout():
     return widgets.Layout(
        border='solid 1px black',
        margin='0px 10px 10px 0px',
        padding='5px 5px 5px 5px'
     )

class mandel3d(widgets.VBox):

    def resolution_update(self, change):
        print('resolution_update','change',change)
        
        plt.ioff()
        self.ax.cla()
        
        self.n=int(change.new)
        self.itNum = np.zeros((self.n,self.n))
        self.Z     = np.zeros((self.n,self.n),dtype=np.complex_)
        print('es la ene',self.n)
        # Make data.
        self.x = np.linspace(-2.25, +0.75, self.n)
        self.y = np.linspace(-1.25, +1.25, self.n)
        self.X, self.Y = np.meshgrid(self.x, self.y, sparse=True)
        #Z = np.nan_to_num(X*Y**2 + 8*X**2*Y**2 + 8*X**3*Y**2 + 3*X**4*Y**2 + 2*Y**4 + 4*X*Y**4 + 3*X**2*Y**4 + Y**6, posinf=255)
    
        for i,vx in enumerate(x):
            for j,vy in enumerate(y):
                self.itNum[i][j], self.Z[i][j] = mandelbrot( vx, vy, self.mI)
    
        # Plot the surface.
        self.surf = self.ax.plot_surface(self.X, self.Y, self.itNum, cmap=cm.coolwarm, linewidth=1, antialiased=True)
    
        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        self.ax.zaxis.set_major_locator(LinearLocator(10))
        self.ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
        # Add a color bar which maps values to colors.
        self.fig.colorbar(self.surf, shrink=0.5, aspect=5)
    
        self.fig.canvas.draw()
        plt.show()
        plt.ion()

    def __init__(self):
        super().__init__()

        output = widgets.Output()      
        with output:
            self.fig = plt.figure()
            self.ax = self.fig.gca(projection='3d')

        self.mI=15
        self.n=30
        self.itNum = np.zeros((n,n))
        self.Z     = np.zeros((n,n),dtype=np.complex_)
        
        self.x = np.linspace(-2.25, +0.75, self.n)
        self.y = np.linspace(-1.25, +1.25, self.n)
        self.X,self.Y = np.meshgrid(self.x, self.y, sparse=True)
        
        for i,vx in enumerate(x):
            for j,vy in enumerate(y):
                self.itNum[i][j], self.Z[i][j] = mandelbrot( vx, vy, self.mI)
        
        # Plot the surface.
        self.surf = self.ax.plot_surface(self.X, self.Y, self.itNum, cmap=cm.coolwarm, linewidth=1, antialiased=True)
    
        # Customize the z axis.
        #ax.set_zlim(-1.01, 1.01)
        self.ax.zaxis.set_major_locator(LinearLocator(10))
        self.ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
        # Add a color bar which maps values to colors.
        self.fig.colorbar(self.surf, shrink=0.5, aspect=5)
    
        #plt.show()
        #self.fig.canvas.draw()
        
        self.resolution_slider = widgets.IntSlider( value=self.n, min=10, max=1000, step=10, description='Pixel resolution²', continuous_update=False )
        self.resolution_slider.observe(self.resolution_update,'value')

        outbox = widgets.Box([output])
        outbox.layout = make_box_layout()
    
        self.resolution_slider.layout = make_box_layout()
    
        self.children = [ self.resolution_slider, outbox ]



# -

mandel3d()

# # repeat

# Hold mouse 1 to rotate  
# Hold mouse 2 to zoom

# +
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(X, Y, itNum, cmap=cm.coolwarm,
                       linewidth=0, antialiased=True)#False
# Customize the z axis.
#ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# +
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.contourf(x, y, itNum, cmap=cm.coolwarm,
                       antialiased=True)#False

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
# -

#type(ax),dir(ax)
plt.isinteractive() #switch: ion() ioff()
ax.can_zoom()

# # HeatMap

from mpl_interactions import  heatmap_slicer #interactive_plot, interactive_plot_factory,
fig,axes = heatmap_slicer( x,y,Z,slices='both',heatmap_names=('mandelbrot'),figsize=(21, 6),labels=('Some wild X variable'),interaction_type='move')

# # Original

# +
fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
# -
# # add collections


# +
 matplotlib.pyplot as plt
from matplotdasdlib import collections, colors, transforms
import numpy as np

nverts = 50
npts = 100

# Make some spirals
r = np.arange(nverts)
theta = np.linspace(0, 2*np.pi, nverts)
xx = r * np.sin(theta)
yy = r * np.cos(theta)
spiral = np.column_stack([xx, yy])

# Fixing random state for reproducibility
rs = np.random.RandomState(19680801)

# Make some offsets
xyo = rs.randn(npts, 2)

# Make a list of colors cycling through the default series.
colors = [colors.to_rgba(c)
          for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.subplots_adjust(top=0.92, left=0.07, right=0.97,
                    hspace=0.3, wspace=0.3)


col = collections.LineCollection([spiral], offsets=xyo,
                                 transOffset=ax1.transData)
trans = fig.dpi_scale_trans + transforms.Affine2D().scale(1.0/72.0)
col.set_transform(trans)  # the points to pixels transform
# Note: the first argument to the collection initializer
# must be a list of sequences of (x, y) tuples; we have only
# one sequence, but we still have to put it in a list.
ax1.add_collection(col, autolim=True)
# autolim=True enables autoscaling.  For collections with
# offsets like this, it is neither efficient nor accurate,
# but it is good enough to generate a plot that you can use
# as a starting point.  If you know beforehand the range of
# x and y that you want to show, it is better to set them
# explicitly, leave out the autolim kwarg (or set it to False),
# and omit the 'ax1.autoscale_view()' call below.

# Make a transform for the line segments such that their size is
# given in points:
col.set_color(colors)

ax1.autoscale_view()  # See comment above, after ax1.add_collection.
ax1.set_title('LineCollection using offsets')


# The same data as above, but fill the curves.
col = collections.PolyCollection([spiral], offsets=xyo,
                                 transOffset=ax2.transData)
trans = transforms.Affine2D().scale(fig.dpi/72.0)
col.set_transform(trans)  # the points to pixels transform
ax2.add_collection(col, autolim=True)
col.set_color(colors)


ax2.autoscale_view()
ax2.set_title('PolyCollection using offsets')

# 7-sided regular polygons

col = collections.RegularPolyCollection(
    7, sizes=np.abs(xx) * 10.0, offsets=xyo, transOffset=ax3.transData)
trans = transforms.Affine2D().scale(fig.dpi / 72.0)
col.set_transform(trans)  # the points to pixels transform
ax3.add_collection(col, autolim=True)
col.set_color(colors)
ax3.autoscale_view()
ax3.set_title('RegularPolyCollection using offsets')


# Simulate a series of ocean current profiles, successively
# offset by 0.1 m/s so that they form what is sometimes called
# a "waterfall" plot or a "stagger" plot.

nverts = 60
ncurves = 20
offs = (0.1, 0.0)

yy = np.linspace(0, 2*np.pi, nverts)
ym = np.max(yy)
xx = (0.2 + (ym - yy) / ym) ** 2 * np.cos(yy - 0.4) * 0.5
segs = []
for i in range(ncurves):
    xxx = xx + 0.02*rs.randn(nverts)
    curve = np.column_stack([xxx, yy * 100])
    segs.append(curve)

col = collections.LineCollection(segs, offsets=offs)
ax4.add_collection(col, autolim=True)
col.set_color(colors)
ax4.autoscale_view()
ax4.set_title('Successive data offsets')
ax4.set_xlabel('Zonal velocity component (m/s)')
ax4.set_ylabel('Depth (m)')
# Reverse the y-axis so depth increases downward
ax4.set_ylim(ax4.get_ylim()[::-1])


plt.show()
