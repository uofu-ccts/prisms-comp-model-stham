import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches

verts = [
    (0., 0.),  # P0
    (2.5,0.0), # P1
    (2.5,5.0), # P2
    (5.0,5.0), # P3
    (5.0,6.0 ),  # P0
	(2.5,6.0), # P1
	(2.5,1.0), # P2
	(0.0,1.0), # P3
	(0.0,0.0),
    ]

codes = [Path.MOVETO,
         Path.CURVE4,
         Path.CURVE4,
         Path.CURVE4,
         Path.LINETO,
         Path.CURVE4,
         Path.CURVE4,
         Path.CURVE4,
         Path.CLOSEPOLY,
         ]

path = Path(verts, codes)



fig,ax = plt.subplots()
ax.set_axis_off()
ax.margins(0.1)
patch = patches.Rectangle((-0.1,-0.1),5.2,6.2,facecolor='white',edgecolor='none')
ax.add_patch(patch)
patch = patches.PathPatch(path, facecolor='grey', alpha=0.3, edgecolor='none')
ax.add_patch(patch)
plt.text(1.0,1.0,"blah");


# xs, ys = zip(*verts)
# ax.plot(xs, ys, 'x--', lw=2, color='black', ms=10)
# 
ax.set_xlim(-0.1,5.1)
ax.set_ylim(-0.1,6.1)

plt.show()
