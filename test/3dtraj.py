
from mpl_toolkits.mplot3d import Axes3D

def threeDtraj(fr):
	fr["start"] = fr["start"] / 60.0
	sublist = fr[fr['actcode'] != 189999]
	print(sublist);
	ax = plt.figure().add_subplot(111,projection='3d')
	ax.plot(fr['lat'],fr['long'],zs=fr['start'],color='k')
	ax.scatter(sublist['lat'],sublist['long'],zs=sublist['start'],c=sublist['start'],s=30)
	plt.show()


ax = plt.figure().add_subplot(111,projection='3d')
ax.plot((fr.iloc[-1].lat, fr.iloc[-1].lat),(fr.iloc[-1].long,fr.iloc[-1].long),zs=(fr.iloc[-1].start,24),color='k')
ax.plot((fr.iloc[-1].lat, fr.iloc[-1].lat),(fr.iloc[-1].long,fr.iloc[-1].long),zs=(0,24),color='r',linestyle=":")
ax.plot(fr['lat'],fr['long'],zs=fr['start'],color='k',linewidth=1.0)
ax.plot(fr['lat'],fr['long'],zs=0.0,color='0.10',linewidth=1.0,linestyle=":")
ax.scatter(sublist['lat'],sublist['long'],zs=sublist['start'],c=sublist['start'],s=50)
ax.set_zlim(0,24.0); ax.set_zticks([0.0,6.0,12.0,18.0,24.0])
ax.set_yticklabels([]); ax.set_xticklabels([])
ax.w_xaxis.set_pane_color((1.0,1.0,1.0,1.0))
ax.w_yaxis.set_pane_color((1.0,1.0,1.0,1.0))
ax.w_zaxis.set_pane_color((1.0,1.0,1.0,1.0))
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Time")
ax.set_aspect(0.5)
ax.view_init(elev=25.0,azim=-60.0)
