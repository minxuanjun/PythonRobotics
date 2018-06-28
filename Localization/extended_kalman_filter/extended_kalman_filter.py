"""

Extended kalman filter (EKF) localization sample

author: Atsushi Sakai (@Atsushi_twi, @minxuanjun) 

Created on Wed Jan 10 16:41:12 2018

"""

import math
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.stats import norm
from scipy.ndimage import gaussian_filter

#Parameter
EXTEND_AREA = 10 # [m] grid map extension lenght
SIM_TIME = 50.0 # simulation time [s]
DT = 0.1 #time tick
MAX_RANGE = 10.0 #maximum observation range
MOTION_STD = 1.0 #standard divation for motion guassion distribution
RANGE_STD = 3.0 #standard deviation for observation gaussion distribution
RADIANS_STD = 40.0
Q = np.diag([RANGE_STD**2, math.radians(RADIANS_STD)**2])

#grid map param
XY_RESO = 0.5 #xy grid resolution
MINX = -15.0
MAXX = 15.0
MINY = -5.0
MAXY = 25.0

#simulation parameters
NOISE_RANGE = 2.0 # [m] 1 sigma range noise parameter
NOISE_SPEED = 0.5 # [m/s] 1 sigma speed noise parameter

show_animation = True

class grid_map():
    def __init__(self):
        self.data = None
        self.xyreso = None
        self.minx = None
        self.maxx = None
        self.miny = None
        self.maxy = None
        self.xw = None
        self.yw = None
        self.dx = 0.0 #movement distance
        self.dy = 0.0 #movement distance
def init_gmap(xyreso, minx, maxx, miny, maxy):
    gmap = grid_map()
    gmap.xyreso = xyreso
    gmap.minx = minx
    gmap.maxx = maxx
    gmap.miny = miny
    gmap.maxy = maxy
    gmap.xw = int(round((gmap.maxx - gmap.minx) / gmap.xyreso))
    gmap.yw = int(round((gmap.maxy - gmap.miny) / gmap.xyreso))
    gmap.data = [[1.0 for i in range(gmap.yw)] for j in range(gmap.xw)]
    gmap = normalize_probability(gmap)
    
    return gmap           
def histogram_filter_localization(gmap, xTrue, u, z, yaw):
    gmap = motion_update(gmap, u, yaw)
    gmap = observation_update(gmap, xTrue, z, RANGE_STD)
    
    return gmap
def calc_gaussian_observation_pdf(gmap, xTrue, z, iz, ix, iy, std):
#predicted range
    x = ix * gmap.xyreso + gmap.minx
    y = iy * gmap.xyreso + gmap.miny
    d = math.sqrt((x - z[iz, 1])**2 + (y - z[iz, 2])**2)
    d_error = d - z[iz, 0]
    radian_observation = np.arctan2(y - z[iz,2], x - z[iz,1])
    radian_mation = np.arctan2(xTrue[1,0] - z[iz,2], xTrue[0,0] -z[iz,1])
    radian_error = radian_observation - radian_mation
    if radian_error > np.pi:
        radian_error -= radian_error
    if radian_error < -np.pi:
        radian_error += np.pi
    z = np.matrix([[d_error],
                   [radian_error]])
    pdf = 1 / np.sqrt(2* np.pi + np.linalg.det(Q)) * np.exp(-1/2 * z.T * np.linalg.inv(Q)*z)
    # likelihood
    #pdf = (1.0 - norm.cdf(abs(d - z[iz, 0]), 0.0, std))
    return pdf

def calc_input():
    v = 1.0 #[m/s]
    yawrate = 0.1 # [rad/s]
    u = np.matrix([v, yawrate]).T
    return u

def motion_model(x,u):
    F = np.matrix([[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, 1.0, 0],
                   [0, 0, 0, 0]])
    B = np.matrix([[DT * math.cos(x[2,0]), 0],
                   [DT * math.sin(x[2,0]), 0],
                   [0.0, DT],
                   [1.0, 0.0]])
    x = F * x + B * u
    
    return x
def observation(xTrue, u, RFID):
    xTrue = motion_model(xTrue,u)
    z = np.matrix(np.zeros((0,3)))
    for i in range(len(RFID[:,0])):
        dx = xTrue[0,0] - RFID[i,0]
        dy = xTrue[1,0] - RFID[i,1]
        d = math.sqrt(dx**2 + dy**2)
        if d <= MAX_RANGE:
            #add noise to range observation
            dn = d + np.random.randn()*NOISE_RANGE
            zi = np.matrix([dn, RFID[i,0], RFID[i,1]])
            z = np.vstack((z,zi))
    
    #add noise to speed
    ud = u.copy()
    ud[0] += np.random.randn() * NOISE_SPEED
    
    return xTrue, z, ud 
def normalize_probability(gmap):
    sump = sum([sum(igmap) for igmap in gmap.data])

    for ix in range(gmap.xw):
        for iy in range(gmap.yw):
            gmap.data[ix][iy] /= sump
    return gmap
def map_shift(gmap, xshift, yshift):
    tgmap = copy.deepcopy(gmap.data)
    
    for ix in range(gmap.xw):
        for iy in range(gmap.yw):
            nix = ix + xshift
            niy = iy + yshift
            
            if nix >= 0 and nix < gmap.xw and niy >= 0 and niy < gmap.yw:
                gmap.data[nix][niy] = tgmap[ix][iy]
    return gmap
def motion_update(gmap, u, yaw):
    gmap.dx += DT * math.cos(yaw) * u[0]
    gmap.dy += DT * math.sin(yaw) * u[0]
    
    xshift = gmap.dx // gmap.xyreso
    yshift = gmap.dy // gmap.xyreso
    # map should be shift
    if abs(xshift) >= 1.0 or abs(yshift) >= 1.0:
        gmap = map_shift(gmap, int(xshift),int(yshift))
        gmap.dx -= xshift * gmap.xyreso
        gmap.dy -= yshift * gmap.xyreso
        
    gmap.data = gaussian_filter(gmap.data, sigma= MOTION_STD)
    
    return gmap
def observation_update(gmap, xTrue, z, std):
    for iz in range(z.shape[0]):
        for ix in range(gmap.xw):
            for iy in range(gmap.yw):
                gmap.data[ix][iy] *= calc_gaussian_observation_pdf(gmap, xTrue, z, iz, ix, iy, std)
    gmap = normalize_probability(gmap)
    
    return gmap
def calc_grid_index(gmap):
    mx, my = np.mgrid[slice(gmap.minx - gmap.xyreso / 2.0, gmap.maxx + gmap.xyreso / 2.0, gmap.xyreso),
                      slice(gmap.miny - gmap.xyreso / 2.0, gmap.maxy + gmap.xyreso / 2.0, gmap.xyreso)]
    return mx, my
    
def draw_heatmap(data, mx, my):
    maxp = max(np.array(data).flatten())
    plt.pcolor(mx, my, data, vmax= maxp, cmap= plt.cm.Blues)
    plt.axis('equal')

def main():
    print(__file__ + 'start!!')
    #RFID position [x, y]
    RFID = np.array([[10.0, 0.0],
                     [10.0, 10.0],
                     [0.0, 15.0],
                     [-5.0, 20.0]])
    time = 0.0
    xTrue = np.matrix(np.zeros((4,1))) # the truth value of the state [x, y, yaw, velocity]
    gmap = init_gmap(XY_RESO,MINX,MAXX, MINY, MAXY)
    mx, my = calc_grid_index(gmap) # for grid map visualization
    trajectory = xTrue.copy()
    
    while SIM_TIME >= time:
        time += DT
        print("Time: ", time)
        u = calc_input()
        yaw = xTrue[2,0]
        xTrue, z, ud = observation(xTrue, u, RFID)
        trajectory = np.hstack((trajectory,xTrue))
        gmap = histogram_filter_localization(gmap, xTrue, u, z, yaw)
        if show_animation:
            plt.cla()
            draw_heatmap(gmap.data, mx, my)
            plt.plot(xTrue[0,:], xTrue[1,:], "xr")
            plt.plot(RFID[:,0], RFID[:, 1], ".k")
            plt.plot(trajectory[0,:].tolist()[0], trajectory[1,:].tolist()[0],'-r')
            for i in range(z.shape[0]):
                plt.plot([xTrue[0,:], z[i,1]], [xTrue[1, :], z[i,2]], "-k")
            plt.title("Time[s]:" + str(time)[0:4])
            plt.pause(0.1)
    print("Done")
if __name__ == '__main__':
    main()
