# CEA TOP AND
# CEA-based Thrust Optimized Parabolic Axisymmetric Nozzle Design
import sys, getopt
sys.path.insert(1, '../CEA')
from pyCEA import cea
import numpy as np
from scipy import interpolate



# RAO angles function
def rao_angles(Lf_querry, Ae_At_querry):
    Ae_At = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 50.0, 100.0])
    Lf =    np.array([0.6, 0.8, 1.0, 1.2])

    Lf, Ae_At = np.meshgrid(Lf,Ae_At)
    Ae_At_f = Ae_At.flatten()
    Lf_f = Lf.flatten()

    ai = [  [20.0, 19.0, 18.0, 16.0],
            [27.0, 22.5, 20.0, 19.0],
            [32.5, 25.5, 22.5, 21.5],
            [34.0, 27.5, 24.0, 23.0],
            [36.0, 29.0, 25.5, 24.5],
            [37.0, 30.5, 27.0, 26.0],
            [37.5, 31.3, 27.5, 26.5],
            [38.0, 32.0, 29.0, 28.0],
            [38.2, 32.1, 29.5, 29.0] ]
    ai = np.array(ai).flatten()

    ae = [  [25.0, 16.0, 11.5, 10.5],
            [20.0, 13.5,  8.7,  6.5],
            [17.2, 11.0,  6.5,  4.5],
            [15.5,  9.5,  6.0,  4.0],
            [14.5,  9.0,  5.6,  3.6],
            [14.0,  8.5,  5.4,  3.4],
            [13.5,  8.0,  5.2,  3.2],
            [13.0,  7.5,  5.0,  3.0],
            [12.7,  7.3,  4.9,  2.9] ]
    ae = np.array(ae).flatten()
    
    points = np.array([Lf_f, Ae_At_f]).transpose()
    aif = interpolate.griddata(points, ai, (Lf_querry, Ae_At_querry), method='cubic' )
    aef = interpolate.griddata(points, ae, (Lf_querry, Ae_At_querry), method='cubic' )

    return aif*np.pi/180, aef*np.pi/180



if __name__=='__main__':
# Input handling
    hstr = 'cta.py -e <Ae/At> -l <Lf> -r <r1/r*>'
    try:
        opts, args = getopt.getopt(sys.argv, "e:l:r:", ["expan=","lf=","r1rs="])
    except getopt.GetoptError:
        print(hstr)
        sys.exit(2)

    ar = 2
    lf = 1
    r1rs = 1

    i = 0
    while i < (len(args)-1)/2:
        i += 1
        opt = args[2*i-1]
        if len(args)>2:
            arg = args[2*i]
        if opt == "-h":
            print(hstr)
            sys.exit()
        elif opt in ("-e","--expan"):
            ar = float(arg)
        elif opt in ("-l","--lf"):
            lf = float(arg)
        elif opt in ("-r","--r1rs"):
            r1rs = float(arg)
    ai, ae = rao_angles(lf, ar)

    Ls = (np.sqrt(ar)-1)/(lf*np.tan(15*np.pi/180))
    Ns = [ r1rs*np.sin(ai), 1+r1rs*(1-np.cos(ai)) ]
    Es = [ Ns[1]+Ls, np.sqrt(ar) ]

    print('L/r*   :   '+'{:9.6f}'.format(Ls))
    print('N/r*   : [ '+'{:9.6f}'.format(Ns[0])+', '+'{:9.6f}'.format(Ns[1])+' ]')
    print('E/r*   : [ '+'{:9.6f}'.format(Es[0])+', '+'{:9.6f}'.format(Es[1])+' ]')
    print('Ai, Ae : [ '+'{:9.6f}'.format(ai*180/np.pi)+', '+'{:9.6f}'.format(ae*180/np.pi)+' ]')
