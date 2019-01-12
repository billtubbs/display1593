import numpy as np
from scipy import spatial
import display1593 as display

print("\n ---------- Colour Testing Script ------------- \n")

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
nx, ny = (8, 32)  # 7 cols, any number of rows
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

# Set default colour intensity scales
#scale = np.array([x**1.5 for x in np.linspace(0, 16, ny)])
rgb_scales = np.zeros((ny, 3))

# Red
rgb_scales[:, 0] = np.array([2.5**x for x in np.linspace(0, 3.9, ny)]) - 1
# Grn
rgb_scales[:, 1] = np.array([2.72**x for x in np.linspace(0, 4.0, ny)]) - 1
# Blu
rgb_scales[:, 2] = np.array([3.0**x for x in np.linspace(0, 4.4, ny)]) - 1

# Need them to be identical at low intensity
rgb_scales[0:9, :-1] = rgb_scales[0:9, 2].reshape(-1,1)

rgb_scales = rgb_scales.round()
print("Colour scales:\n", rgb_scales.__repr__())

# Calculate LED colours
#led_cols = col_bars[led_data[:, 0]]*led_data[:, 1].reshape(-1, 1)
led_cols = col_bars[led_data[:, 0]]*rgb_scales[led_data[:, 1]]

# Set LED colours
dis.setAllLeds(led_cols)
