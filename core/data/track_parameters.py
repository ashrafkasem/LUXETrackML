import numpy as np
from scipy import optimize
from math import acos, asin, atan, pi
from numpy import cos, sin
import sys

#from https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
def fitcircle(x,y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    
    # linear system defining the center (uc, vc) in reduced coordinates:
    #    Suu * uc +  Suv * vc = (Suuu + Suvv)/2
    #    Suv * uc +  Svv * vc = (Suuv + Svvv)/2
    Suv  = np.sum(u*v)
    Suu  = np.sum(u**2)
    Svv  = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)
    
    # Solving the linear system
    A = np.array([ [ Suu, Suv ], [Suv, Svv]])
    B = np.array([ Suuu + Suvv, Svvv + Suuv ])/2.0
    uc, vc = np.linalg.solve(A, B)

    xc = x_m + uc
    yc = y_m + vc

    # Calcul des distances au centre (xc, yc)
    Ri_1     = np.sqrt((x-xc)**2 + (y-yc)**2)
    R      = np.mean(Ri_1)
    #residu_1 = sum((Ri_1-R_1)**2)
    
    return xc, yc, R, GetSign(x,y,xc,yc)

def calc_R(x, y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f_2b(x,y, c):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x,y,*c)
    return Ri - Ri.mean()

def Df_2b(x, y, c):
    """ Jacobian of f_2b
    The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
    xc, yc     = c
    df2b_dc    = np.empty((len(c), x.size))

    Ri = calc_R(xc, yc)
    df2b_dc[0] = (xc - x)/Ri                   # dR/dxc
    df2b_dc[1] = (yc - y)/Ri                   # dR/dyc
    df2b_dc    = df2b_dc - df2b_dc.mean(axis=1)[:, np.newaxis]

    return df2b_dc

def fit2circle(x,y):
    center_estimate = np.mean(x), np.mean(y)
    center, ier = optimize.leastsq(f_2b, center_estimate, Dfun=Df_2b, col_deriv=True)
    xc, yc = center
    Ri       = calc_R(*center)
    R        = Ri.mean()
    #residu_2   = sum((Ri - R)**2)
    #check direction of the points:
    q = GetSign(x,y,xc,yc)
    return xc, yc, R, q

def GetIP(xc,yc,R):
    xIP, yIP = xc*(1 - np.abs(R)/np.sqrt(xc**2 + yc**2)), yc*(1 - np.abs(R)/np.sqrt(xc**2 + yc**2))
    phi0 = np.arctan2(yc,xc)
    d0 = np.sqrt(xc**2 + yc**2) - R # d0 has a sign
    return xIP, yIP, d0, phi0

def GetSign(x,y,xc,yc):
    phi = np.unwrap(np.arctan2(y-yc, x-xc))
    q = -(np.sign( np.mean( np.unwrap(phi[1:]-phi[:-1])))) #clockwise - decreasing phi: possitive charge
    return q    

def FitTrackThreePoints(x,y,z):
    #fit {xi,yi} to circle: (x-xc)**2+(y-yc)**2 = rho**2
    dx10 = x[1]-x[0]; dx21=x[2]-x[1]; dy10 = y[1]-y[0]; dy21=y[2]-y[1]
    A = np.array([ [ dx10, dy10 ], [dx21, dy21]]) * 2.0
    B = np.array([ dx10 * (x[1]+x[0]) + dy10*(y[1]+y[0]), dx21 * (x[2]+x[1]) + dy21 * (y[2]+y[1]) ])
    xc, yc = np.linalg.solve(A, B) #solve A*[xc,yc]=B
    rho = np.mean(np.sqrt((x-xc)**2 + (y-yc)**2))
    d0 = np.sqrt(xc**2 + yc**2) - rho
    phi0 = np.arctan2(yc, xc)
    #fit {zi,ri=sqrt(xi**2+yi**2)} to a streight line: z = z0 - r*tanL
    r = np.sqrt(x**2+y**2)
    (tanL,z0) = np.polyfit(-r,z,1)
    return rho,phi0,d0,z0,tanL

def FitTrackTwoPoints(x,y,z):
    '''
    circle that cross the origin, with two points v1=(x1,y1) and v2=(x2,y2) obeys:
    dr = 2*rho*sin(dphi)
    where dr=sqrt(dx**2+dy**2), and dphi is the angle between v1,v2
    '''
    phi = np.arctan2(y,x)
    dphi=acos(cos(phi[1]-phi[0]))      
    dr = np.sqrt((x[1]-x[0])**2+(y[1]-y[0])**2)
    rho =0.5*dr/dphi
    
    '''
    equation of particle propogation in z direction obeys:
    z = z0 - rho*phi*tan(lambda) ~ r*tan(lambda) (from rho*sin(phi/2)~rho*phi/2=r/2)
    where phi is the angle betwen v0 and vi, where v0 pointing to the origin
    then one can write for two points (z1,phi1), (z2,phi2)
    z2 - z1 = -rho(phi2-phi1)tan(lambda) = dr*tan(lambda)
    z0 = zi + ri*tan(lambda)
    '''
    r = np.sqrt(x**2+y**2)
    dz = z[1] - z[0]
    
    tanL = -dz/dr
    z0 = np.mean(z + r*tanL)

    return rho,z0,tanL

def TrackParameters(x,y,z):
    if len(x)<3:
        return 0,0,0,0,0,0
    xc,yc,r,q = fitcircle(x,y)
    phi0 = np.arctan2(yc,xc)
    d0 = np.sqrt(xc**2+yc**2)-r
    phi = np.unwrap(np.arctan2(yc - y, xc - x) - phi0)
    tanL, z0 = np.polyfit(phi*r,z,1)
    return d0,z0,r,phi0,tanL,q

def add_parameters(grp):
    x,y,z = grp['tx'],grp['ty'],grp['tz']
    d0,z0,r,phi0,tanL, q = TrackParameters(x,y,z)
    grp['q'] = q
    grp['z0'] = z0
    grp['d0'] = d0
    grp['rho'] = r
    grp['ntrk'] = len(x)
    return grp

def printStatus(i,n):
    point = n // 100
    increment = n / 75
    if(i % (point) == 0):
        sys.stdout.write("\r[" + "=" * int(i / increment) +  " " * int((n - i)/ increment) + "]" +  str(i / point) + "%")
        sys.stdout.flush()
    if i==(n-1):
        sys.stdount.write('\n')
        