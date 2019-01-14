#!/usr/bin/python
# Python script to generate a set of LED intensity
# scales to correct for non-linearity in LED colour
# intensities.  Displays a test screen to allow
# user to adjust scales until they look balanced and
# linear.

import numpy as np
from scipy import spatial

print("\n ---------- Colour Testing Script ------------- \n")

# Create RGB colour intensity scales
# Adjust the constants in code below to balance colours
n_levels = 32
rgb_scales = np.zeros((n_levels, 3))

# Red
rgb_scales[:, 0] = np.array(
    [2.5**x for x in np.linspace(0, 4.2, n_levels)]
) - 1

# Green
rgb_scales[:, 1] = np.array(
    [2.72**x for x in np.linspace(0, 3.9, n_levels)]
) - 1

# Blue
rgb_scales[:, 2] = np.array(
    [2.85**x for x in np.linspace(0, 4.2, n_levels)]
) - 1

# Need them to be identical at low intensity levels
for i, rgb in enumerate(rgb_scales):
    avg = rgb.mean()
    if avg > 2.5 or i>10:
        break
    rgb_scales[i, :] = avg

# Save a set of arrays
print("Saving all RGB colour scales to file...")
all_rgb_scales = np.swapaxes(np.stack([rgb_scales]*8, axis=2), 0, 2)
all_rgb_scales = np.swapaxes(all_rgb_scales, 1, 2)

# Adjust each scale for different brightness settings
for b in range(1, 8):
    all_rgb_scales[b, :, :] *= 1.15**b

assert np.all(all_rgb_scales.round() < 256)

# Convert to integers
all_rgb_scales = all_rgb_scales.round().astype('uint8')

# Save file to data folder for use by display1593.py
filename = 'data/rgb_scales.npy'
np.save(filename, all_rgb_scales)
print("Array size %s saved to file %s" % (all_rgb_scales.shape, filename))

# Now load display module and connect to display
import display1593 as display

# Brightness (1-8)
brightness = 2

# Connect to display
dis = display.Display1593()
dis.connect()
dis.clear()

numCells = display.leds.numCells

# Retrieve led co-ordinates
x = display.leds.centres_x
y = display.leds.centres_y
dis_pts = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1)

# Plan for displaying colour grid
col_bars = np.array([
    (1, 1, 1),
    (0, 0, 0),
    (1, 0, 0),
    (1, 1, 0),
    (0, 1, 0),
    (0, 1, 1),
    (0, 0, 1),
    (1, 0, 1)
])

# Define size of colour grid
nx, ny = (8, n_levels)  # 7 cols, any number of rows
xx, yy = np.meshgrid(range(nx), range(ny))
xx = xx.ravel()
yy = yy.ravel()

# Grid index
pts = np.array(list(zip(xx, yy)))

# Physical co-ordinate of centre-points
c_pts = np.zeros_like(pts)
c_pts[:, 0] = (pts[:, 0] + 0.5)*display.leds.width/nx
c_pts[:, 1] = (pts[:, 1] + 0.5)*display.leds.height/ny

# KDTree to find nearest centre-point
tree = spatial.KDTree(c_pts)

try:
    led_data=np.load('led_data.npy')
except:
    print("Generating colour map data...")
    led_data = np.empty((numCells, 2), dtype='i8')
    for i in range(numCells):
        p = dis_pts[i]
        q = tree.query(p)
        gp = pts[q[1]]
        led_data[i, :] = gp
        #print(i, p, q[1], gp)
    np.save('led_data.npy', led_data)

assert 0 < brightness < 9

print("Brightness:", brightness)
print("Colour scale:\n", all_rgb_scales[brightness].__repr__())

loop = True
while loop:

    # Calculate LED colours
    led_cols = col_bars[led_data[:, 0]]* \
        all_rgb_scales[brightness][led_data[:, 1]]

    # Set LED colours
    print("Displaying colour chart...")
    dis.setAllLeds(led_cols)
    r, g, b = led_cols[:, 0], led_cols[:, 1], led_cols[:, 2]
    print("RGB ranges: (%d - %d), (%d - %d), (%d - %d)" %
          (r.min(), r.max(), g.min(), g.max(), b.min(), b.max()))

    while True:
        s = input("Change brightness (0-7) or press 'q' to quit:")
        try:
            brightness = int(s)
            if not 0 <= brightness < 8:
                raise ValueError
            break
        except ValueError:
            if s.lower() == 'q':
                loop = False
                break
            print("Try again")
