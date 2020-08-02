#!/usr/bin/env python
# coding: utf-8

# # **import** Mandelbrot Cython, ipywidget with ipympl matplotlib

# In[1]:


get_ipython().run_line_magic('load_ext', 'Cython')
get_ipython().run_line_magic('matplotlib', 'widget')
import numpy as np
import ipywidgets as widgets
from ipywidgets import Layout
import matplotlib.pyplot as plt


# ## Mandelbrot called from C(ython)

# In[2]:


get_ipython().run_cell_magic('cython', '', 'cpdef Mandelbrot(int[:,::1] Z, float xl, float xr, float yd, float yu, int Rx, int Ry, int maxIter ) : #except *:\n    cdef int y, x, n\n    cdef complex z, c\n    cdef float dx=Rx/(xr-xl)\n    cdef float dy=Ry/(yu-yd)\n    for y in range(Ry):\n        for x in range(Rx):\n            c = xl + x / dx + 1j*(yu - y / dy)\n            z = 0\n            for n in range(maxIter):\n                if z.real**2 + z.imag**2 >= 4:\n                    break\n                z = z*z + c\n            Z[y, x] = n')


# ## User interface, observe events

# In[3]:


output = widgets.Output()
with output:
    fig = plt.figure()
    fig.canvas.toolbar_position = 'left'
    fig.canvas.header_visible = False
    #fig.canvas.capture_scroll = True
    #fig.canvas.footer_visible = False
    ax = fig.add_subplot(111)
    ax.set_xlabel('Real numbers')
    ax.set_ylabel('Imaginary numbers')
    plt.tight_layout()

# create some intial data
xl, xr = -2.0, 0.66
yd, yu = -1.4, 1.4
Rx, Ry = 302, 302
maxIter= 200

def redraw(xl, xr, yd, yu, Rx, Ry, maxIter, output):
    with output:
        Z=np.zeros((Ry,Rx), dtype=np.int32)
        Mandelbrot(Z, xl, xr, yd, yu, Rx, Ry, maxIter)
        xoffset= (xr-xl)/(2*Rx)
        yoffset= (yu-yd)/(2*Ry)
        # CLEAR AXES
        ax.cla()
        im = ax.imshow(Z, interpolation='none', aspect='equal',
                       origin='upper',
                       extent=[xl+xoffset, xr+xoffset, yd+yoffset, yu+yoffset] )
                        # check https://matplotlib.org/3.2.1/tutorials/intermediate/imshow_extent.html#
        ax.set_xlabel('Real numbers')
        ax.set_ylabel('Imaginary numbers')
        return "MandelPlot "+"{:,}".format(Z.shape[0]*Z.shape[1])+" pixels calculated"

# create some control elements
xrange_picker = widgets.FloatRangeSlider(value=(xl,xr), min=-2.0, max=2.0, step=0.001, readout_format='.4f', description='real', 
                                         continuous_update=False, orientation='vertical', layout=Layout( height='90%', width='110%'))
yrange_picker = widgets.FloatRangeSlider(value=(yd,yu), min=-1.8, max=1.8, step=0.001, readout_format='.4f', description='imaginary', 
                                         continuous_update=False, orientation='vertical', layout=Layout( height='90%', width='110%'))
Rx_picker = widgets.IntText(value=302, min=2, max=10002, step=50, description='Real grid qty.', 
                            continuous_update=False, layout=Layout( width='90%'))
Ry_picker = widgets.IntText(value=302, min=2, max=10002, step=50, description='Imaginary qty.', 
                            continuous_update=False, layout=Layout( width='90%'))
maxIter_picker = widgets.IntText(value=200, min=1, max=1000, step=50, description='Iterations',
                                 continuous_update=False, layout=Layout( width='90%'))
reset_button = widgets.Button(description='Reset values!',
                              continuous_update=False)
title_Text = widgets.Label(layout=Layout( align_self='center' ))

# callback functions
def update(change):
    xl, xr = xrange_picker.value
    yd, yu = yrange_picker.value
    Rx, Ry = Rx_picker.value, Ry_picker.value
    maxIter= maxIter_picker.value
    title_Text.value = redraw( xl, xr, yd, yu, Rx, Ry, maxIter, output)
    
def button_reset_clicked(change):
    xl, xr = -2.0, 0.66
    yd, yu = -1.4, 1.4
    xrange_picker.value=(xl,xr)
    yrange_picker.value=(yd,yu)
    
    Rx, Ry = 302, 302
    Rx_picker.value, Ry_picker.value = Rx, Ry
    
    maxIter=200
    maxIter_picker.value=maxIter
    
    title_Text.value = redraw( xl, xr, yd, yu, Rx, Ry, maxIter, output)
    
# connect callbacks and traits
xrange_picker.observe(update, 'value')
yrange_picker.observe(update, 'value')
Rx_picker.observe(update, 'value')
Ry_picker.observe(update, 'value')
maxIter_picker.observe(update, 'value')
reset_button.on_click(button_reset_clicked)
    
controls = widgets.HBox([widgets.HBox([xrange_picker, yrange_picker], layout=Layout( height='100%', width='101%')),
                         widgets.VBox([title_Text, Rx_picker, Ry_picker, maxIter_picker,reset_button], layout=Layout( height='100%', width='99%'))
                        ])

title_Text.value = redraw(xl, xr, yd, yu, Rx, Ry, maxIter, output)
widgets.HBox([output, controls])


# ## User interface, observe events, encapsulate

# ### ***box layout***

# In[4]:


def make_box_layout():
     return widgets.Layout(
        border='solid 1px black',
        margin='0px 10px 10px 0px',
        padding='5px 5px 5px 5px'
     )


# ### Iterable widget object

# #### Layout keywords  
# 
# justify-content : space-around  
#     align-items : center  
#   align-content : space-around  
# 
# from ipywidgets import AppLayout
#   AppLayout(
#     center=fig.canvas,
#     footer=slider,
#     pane_heights=[0, 6, 1]
# )
# 
# #### AppLayout Object  
# 
# AppLayout(center=m, 
#           header=header,
#           left_sidebar=VBox([Label("Basemap:"),
#                              basemap_selector,
#                              Label("Overlay:"),
#                              heatmap_selector]),
#           right_sidebar=fig,
#           footer=out,
#           pane_widths=['80px', 1, 1],
#           pane_heights=['80px', 4, 1],
#           height='600px',
#           grid_gap="30px")
class FdoLogger()
    self.CurrentLog = 'Fdo start'
    print('Fdo start')

    def logginHere(logginString, self_textArea_value):
        if len(self_textArea_value):
            l='update slider %s->%s\n'%(change.old,change.new)
            print(logginHere, end=', ')
        return logginString+self.textArea.value
# In[5]:


class Sins(widgets.HBox):
    
    def __init__(self):
        super().__init__()
        output = widgets.Output()

        self.initial_color = '#FF00DD'
        self.initial_freak = 2
        self.x = np.linspace(0, 2 * np.pi, 100)

        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(5, 3.5))
            self.line, = self.ax.plot(self.x, np.sin(self.x), self.initial_color)

            self.fig.canvas.toolbar_position = 'left'
            self.ax.grid(True)

            # define widgets
            self.intSlider = widgets.IntSlider( value=self.initial_freak, min=0, max=10, step=1, description='Superfreak')#, continuous_update=False)
            self.resetButton = widgets.Button( description='Reset values!')
            self.titleText = widgets.Label( value='Reset values!', layout=Layout( align_self='center') )
            self.textArea = widgets.Textarea(  placeholder='Type something',  disabled=False )#,value='Hello World',description='Resultado:',
                                               # layout=Layout( max_width='500px'))#value='Reset values!', layout=Layout( align_self='center' ))

        # layout
        controls = widgets.VBox([ self.titleText, self.intSlider, self.resetButton, self.textArea]) #, layout=Layout( width='100%')
        controls.layout = make_box_layout()
        
        out_box = widgets.Box([output],layout=Layout( width='100%'))
        output.layout = make_box_layout()
        
        # observe stuff
        self.intSlider.observe(self.update, 'value')
        self.resetButton.on_click(self.button_reset_pressed)
        
         # add to children
        self.children = [controls, output]#out_box

    def destroy(self):
        self.fig.clf()
    
    def update(self, change):
        """Draw line in plot"""
        logginHere='update slider %s->%s\n'%(change.old,change.new)
        print(logginHere, end=', ')
        self.textArea.value=logginHere+self.textArea.value[:500]
        self.line.set_ydata(np.sin(change.new * self.x))
        self.fig.canvas.draw()
        
    def button_reset_pressed(self, change):
        #self.ax.cla()
        logginHere='reset button %s->%s'%(self.intSlider.value,self.initial_freak)
        print(logginHere, end=', ')
        self.textArea.value=logginHere#+self.textArea.value[:500]
        self.intSlider.value=self.initial_freak
        self.line.set_ydata(np.sin(self.initial_freak * self.x))
            
        


# In[7]:


a=Sins()
a


# In[8]:


from matplotlib.backend_bases import Event
from ipympl.backend_nbagg import Toolbar

#HOME
home = Toolbar.home

def new_home(self, *args, **kwargs):
    s = 'home_event'
    event = Event(s, self)
    event.foo = 0
    self.canvas.callbacks.process(s, event)
    home(self, *args, **kwargs)

Toolbar.home = new_home

def handle_home(evt):
    print('new home ',evt.foo)

#ZOOM
zoom = Toolbar.zoom

def new_zoom(self, *args, **kwargs):
    s = 'zoom_event'
    event = Event(s, self)
    event.foo = 1
    self.canvas.callbacks.process(s, event)
    zoom(self, *args, **kwargs)

Toolbar.zoom = new_zoom

def handle_zoom(evt):
    print('new zoom ' , evt.foo, ax.get_xlim(), ax.get_ylim())

#PRESS
press_zoom = Toolbar.press_zoom

def new_press_zoom(self, *args, **kwargs):
    s = 'press_zoom_event'
    event = Event(s, self)
    event.foo = 2
    self.canvas.callbacks.process(s, event)
    press_zoom(self, *args, **kwargs)

Toolbar.press_zoom = new_press_zoom

def handle_press_zoom(evt):
    print('new press_zoom ' , evt.foo, ax.get_xlim(), ax.get_ylim())

#release
release_zoom = Toolbar.release_zoom

def new_release_zoom(self, *args, **kwargs):
    s = 'release_zoom_event'
    event = Event(s, self)
    event.foo = 3
    self.canvas.callbacks.process(s, event)
    release_zoom(self, *args, **kwargs)

Toolbar.release_zoom = new_release_zoom

def handle_release_zoom(evt):
    print('new release_zoom ' , evt.foo, ax.get_xlim(), ax.get_ylim())

def onclick(event):
    print('%s click: button=%d, x=%d, y=%d, xdata=%.3f, ydata=%.3f, name=%s' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata, event.name))    

fig, ax = plt.subplots()

fig.canvas.mpl_connect('home_event', handle_home)
fig.canvas.mpl_connect('zoom_event', handle_zoom)
fig.canvas.mpl_connect('press_zoom_event', handle_press_zoom)
fig.canvas.mpl_connect('release_zoom_event', handle_release_zoom)
cidPress   = fig.canvas.mpl_connect('button_press_event'  , onclick)
cidDelease = fig.canvas.mpl_connect('button_release_event', onclick)

ax.plot(np.random.rand(10))
plt.text(0.35, 0.5, 'Hello world!', dict(size=30))
plt.show()


# # Inspired by
# https://kapernikov.com/ipywidgets-with-matplotlib/

# In[9]:


output = widgets.Output()

# create some x data
x = np.linspace(0, 2 * np.pi, 100)

# default line color
initial_color = '#FF00DD'

with output:
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 4))
    # move the toolbar to the bottom
fig.canvas.toolbar_position = 'bottom'
ax.grid(True)    
line, = ax.plot(x, np.sin(x), initial_color)

# create some control elements
int_slider = widgets.IntSlider(value=1, min=0, max=10, step=1, description='freq')
color_picker = widgets.ColorPicker(value=initial_color, description='pick a color')
text_xlabel = widgets.Text(value='', description='xlabel', continuous_update=False)
text_ylabel = widgets.Text(value='', description='ylabel', continuous_update=False)
reset_button = widgets.Button(description="Reset values!", layout=Layout(align_self='flex-end')) #center flex-end flex-start
#reset_button.layout.align_self='center'

# callback functions
def update(change):
    """redraw line (update plot)"""
    line.set_ydata(np.sin(change.new * x))
    fig.canvas.draw()
    
def line_color(change):
    """set line color"""
    line.set_color(change.new)
    
def update_xlabel(change):
    ax.set_xlabel(change.new)
    
def update_ylabel(change):
    ax.set_ylabel(change.new)
    
def on_button_clicked(wtf):
    text_xlabel.value = 'x'
    text_ylabel.value = 'y'
    color_picker.value= initial_color
    int_slider.value = 1
    with output:
        print('yey!')

# connect callbacks and traits
int_slider.observe(update, 'value')
color_picker.observe(line_color, 'value')
text_xlabel.observe(update_xlabel, 'value')
text_ylabel.observe(update_ylabel, 'value')
reset_button.on_click(on_button_clicked)

text_xlabel.value = 'x'
text_ylabel.value = 'y'

controls = widgets.VBox([int_slider, color_picker, text_xlabel, text_ylabel, reset_button])
widgets.HBox([controls, output])


# ## Encapsulated/class version

# In[10]:


def make_box_layout():
     return widgets.Layout(
        border='solid 1px black',
        margin='0px 10px 10px 0px',
        padding='5px 5px 5px 5px'
     )

class Sines(widgets.HBox):
    
    def __init__(self):
        super().__init__()
        output = widgets.Output()

        self.x = np.linspace(0, 2 * np.pi, 100)
        initial_color = '#FF00DD'

        with output:
            self.fig, self.ax = plt.subplots(constrained_layout=True, figsize=(5, 3.5))
        self.line, = self.ax.plot(self.x, np.sin(self.x), initial_color)
        
        self.fig.canvas.toolbar_position = 'bottom'
        self.ax.grid(True)

        # define widgets
        int_slider = widgets.IntSlider(
            value=1, 
            min=0, 
            max=10, 
            step=1, 
            description='freq'
        )
        color_picker = widgets.ColorPicker(
            value=initial_color, 
            description='pick a color'
        )
        text_xlabel = widgets.Text(
            value='', 
            description='xlabel', 
            continuous_update=False
        )
        text_ylabel = widgets.Text(
            value='', 
            description='ylabel', 
            continuous_update=False
        )

        controls = widgets.VBox([
            int_slider, 
            color_picker, 
            text_xlabel, 
            text_ylabel
        ])
        controls.layout = make_box_layout()
        
        out_box = widgets.Box([output])
        output.layout = make_box_layout()

        # observe stuff
        int_slider.observe(self.update, 'value')
        color_picker.observe(self.line_color, 'value')
        text_xlabel.observe(self.update_xlabel, 'value')
        text_ylabel.observe(self.update_ylabel, 'value')
        
        text_xlabel.value = 'x'
        text_ylabel.value = 'y'
        
        # add to children
        self.children = [controls, output]
    
    def update(self, change):
        """Draw line in plot"""
        self.line.set_ydata(np.sin(change.new * self.x))
        self.fig.canvas.draw()

    def line_color(self, change):
        self.line.set_color(change.new)

    def update_xlabel(self, change):
        self.ax.set_xlabel(change.new)

    def update_ylabel(self, change):
        self.ax.set_ylabel(change.new)
        
        
#Sines()


# In[11]:


Sines()


# # Yet another mandelbrot calculation using C
# From Shaded & power normalized rendering:  
# https://matplotlib.org/3.2.1/gallery/showcase/mandelbrot.html

# In[12]:


#%load_ext Cython
import numpy as np


# In[13]:


#cimport numpy as np
#%%cython -a
cpdef mandelbrot_set(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0):
    
    #int[:,::1] Z
    #cdef int [:] foo 
    #NO! cdef int [xmin:xmax:xn] X
    #cdef int [:] X
    #cdef int [:] Y
    X = np.linspace(xmin, xmax, xn).astype(np.float32)
    Y = np.linspace(ymin, ymax, yn).astype(np.float32)
    C = X + Y[:, None] * 1j
    N = np.zeros_like(C, dtype=int)
    Z = np.zeros_like(C)
    for n in range(maxiter):
        I = abs(Z) < horizon
        N[I] = n
        Z[I] = Z[I]**2 + C[I]
    N[N == maxiter-1] = 0
    return Z, N


# In[ ]:


import numpy as np
np.linspace(-3, 3, 6).astype(np.float32)


# # sobras

# In[ ]:





# In[ ]:


widgets.Textarea(
    '\n'.join([w for w in dir(widgets) if not w.islower()]),
    layout=widgets.Layout(height='200px')
)


# In[ ]:


a=widgets.Text(value='hola', description='chao')


# In[ ]:


get_ipython().run_cell_magic('html', '', '<style>\n.mytext > .widget-label {\n    font-style: italic;\n    color: blue;\n    font-size: 30px;\n}\n.mytext > input[type="text"] {\n    font-size: 20px;\n    color: red;\n}\n</style>')


# In[ ]:


a.add_class("mytext")
a

