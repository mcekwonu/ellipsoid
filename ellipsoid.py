import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from numpy import linalg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def plotEllipsoid(A, center):
	"""Plot Ellipsoid with A matrix and center"""
	# random select color
	color = tuple(np.random.randint(256, size=3) / 255)
	
	# find the rotation matrix and radii of the axes
	U, s, rotation = linalg.svd(A)
	radii = 1.0 / np.sqrt(s)
	
	u = np.linspace(0, 2 * np.pi, 100)
	v = np.linspace(0, np.pi, 100)
	
	x = radii[0] * np.outer(np.cos(u), np.sin(v))
	y = radii[1] * np.outer(np.sin(u), np.sin(v))
	z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
	
	for i, _ in enumerate(x):
		for j, _ in enumerate(x):
			[x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center
	
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color=color, alpha=0.4)
	plt.show()
	
	
class Ellipsoid:
	"""Class to make ellipsoid."""
	
	@staticmethod
	def getMinVolEllipse(P, tolerance=0.01):
		"""Find the minimum volume of ellipsoid which holds all the points.
		
		based on code from https://github.com/minillinim/ellipsoid and modified.
		
		Parameters:
			P: numpy array of N-dimensional points like this:
				P = [[x, y, z, ...], <--- one point per line
					 [x, y, z, ...],
					 [x, y, z, ...]]
			
			tolerance: convergence criterion
		
		Returns:
			(center, radii, rotation)
		"""
		N, d = np.shape(P)
		d = float(d)
		
		# Q will the working array
		Q = np.vstack([np.copy(P.T), np.ones(N)])
		QT = Q.T
		
		# initializations
		err = 1.0 + tolerance
		u = (1.0 / N) * np.ones(N)
		
		# Using Khachiyan Algorithm
		while err > tolerance:
			V = np.dot(Q, np.dot(np.diag(u), QT))
			M = np.diag(np.dot(QT, np.dot(linalg.inv(V), Q)))
			k = np.argmax(M)
			maximum = M[k]
			step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
			u_n = (1 - step_size) * u
			u_n[k] += step_size
			err = linalg.norm(u_n - u)
			u = u_n
		
		# center of the ellipse
		center = np.dot(P.T, u)
		
		# A matrix for the ellipse
		A = linalg.inv(np.dot(P.T, np.dot(np.diag(u), P)) -
		               np.array([[a * b for b in center] for a in center])) / d
		
		# Get the radii and rotation
		U, s, rotation = linalg.svd(A)
		radii = 1 / np.sqrt(s)
		
		return center, radii, rotation
	
	@staticmethod
	def getEllipsoidVolume(radii):
		"""Calculate the volume of the blob"""
		return 4/3 * np.pi *radii[0]*radii[1]*radii[2]
		
	@staticmethod
	def plot_ellipsoid(center, radii, rotation, ax=None, plot_axes=False, showSurface=False, alpha=0.5):
		"""Plot an ellipsoid"""
		color = tuple(np.random.randint(256, size=3) / 255)
		
		make_ax = ax is None
		if make_ax:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			
		u = np.linspace(0, 2*np.pi, 100)
		v = np.linspace(0, np.pi, 100)
		
		# get cartesian coordinates corresponding to the spherical angles:
		x = radii[0] * np.outer(np.cos(u), np.sin(v))
		y = radii[1] * np.outer(np.sin(u), np.sin(v))
		z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
		
		# rotate accordingly
		for i, _ in enumerate(x):
			for j, _ in enumerate(x):
				[x[i, j], y[i, j], z[i, j]] = np.dot([x[i, j], y[i, j], z[i, j]], rotation) + center
				
		if plot_axes:
			axes = np.array([[radii[0], 0, 0],
			                 [0, radii[1], 0],
			                 [0, 0, radii[2]]])
			# rotate accordingly
			for i, axis in enumerate(axes):
				axes[i] = np.dot(axes[i], rotation)
				
			# plot axes
			for p in axes:
				X = np.linspace(-p[0], p[0], 100) + center[0]
				Y = np.linspace(-p[1], p[1], 100) + center[1]
				Z = np.linspace(-p[2], p[2], 100) + center[2]
				ax.plot(X, Y, Z, color=color)
		
		# Plot ellipsoid
		if showSurface:
			ax.plot_surface(x, y, z, rstride=3, cstride=3, color=color, alpha=alpha)
		else:
			ax.plot_wireframe(x, y, z, rstride=3, cstride=3, color=color, alpha=alpha)
		
		if make_ax:
			plt.show()
			
			
if __name__ == '__main__':
	P = np.reshape([np.random.random()*100 for i in range(300)], (-1, 3))
	
	# find the ellipsoid
	c, r, rot = Ellipsoid().getMinVolEllipse(P, 0.01)
	
	# Plot ellipsoid
	Ellipsoid().plot_ellipsoid(c, r, rot, plot_axes=True)
	
	# with known A matrix and center, the plotEllipsoid function is used directly
	# plot ellipsoid given the A matrix and center
	A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 2]])
	center = [0, 0, 0]
	plotEllipsoid(A, center)
