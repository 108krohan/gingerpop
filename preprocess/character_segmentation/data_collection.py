import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QPushButton
import sys
import os
from time import sleep

plt.switch_backend("Qt5Agg")
plt.plot([1,2,3,4,5])
plt.show()

#filename="trial1.png"

try:
    filename = sys.argv[1]
    print("Using file :", filename)
    filename, extension = filename.split('.')
except:
    print("PLEASE SPECIFY PROPER ARGUMENTS WHILE RUNNING FILE")
    sys.exit(1)

try:
    counter = int(sys.argv[2])
except:
    counter = 0

print("STARTING COUNT OF FILE FROM", counter)

def check_save(ax):
    sleep(0.5)
    global x_lims
    global y_lims
    global counter
    if x_lims and y_lims:
        print(x_lims, y_lims)
        x_lims = list(x_lims)
        y_lims = list(y_lims)
        image = im.copy()
        if x_lims[0] > x_lims[1]:
            x_lims[0], x_lims[1] = x_lims[1], x_lims[0]
        if y_lims[0] > y_lims[1]:
            y_lims[0], y_lims[1] = y_lims[1], y_lims[0]
        if x_lims[0]<0:
            x_lims[0] = 0
        if y_lims[0]<0:
            y_lims[0] = 0
        image = image[int(np.ceil(y_lims[0])):int(np.ceil(y_lims[1])), int(np.ceil(x_lims[0])):int(np.ceil(x_lims[1]))]
        #print("SAVING", filename+"_"+str(counter)+"."+extension)
        plt.imsave(arr = image, fname = filename+"_"+str(counter)+"."+extension)
        os.system("python3 qt_char_select.py " + "\"" + filename+"_"+str(counter) + "." + extension + "\"")
        counter += 1
    else:
        print("Please select a character before ML data saving")

fig = plt.figure()
ax = fig.add_subplot(111)

# Define ML widget
wid = QPushButton("ML Data : Click Here")
wid.clicked.connect(check_save)

toolbar = fig.canvas.toolbar
toolbar.toolitems = toolbar.toolitems[0]
toolbar.addWidget(wid)


y_lims = None
x_lims = None

# Declare and register callbacks
def on_xlims_change(axes):
    global x_lims
    x_lims = axes.get_xlim()

def on_ylims_change(axes):
    global y_lims
    y_lims = axes.get_ylim()

ax.callbacks.connect('xlim_changed', on_xlims_change)
ax.callbacks.connect('ylim_changed', on_ylims_change)

im = plt.imread(filename+"."+extension)
print(im.shape)
arr = np.asarray(im)
plt_image = ax.imshow(arr)
plt.show()

