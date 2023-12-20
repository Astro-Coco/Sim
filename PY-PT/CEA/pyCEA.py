from os import system, remove, chdir
from random import randint
from pathlib import Path
import getopt, sys, json

"""
pyCEA is a easy to install version of rocketCEA. It wraps NASA's CEA code, written in fortran.

TO RUN, add the following lines at the top of your code (replace pyCEA_path)

import sys
sys.path.insert(1, 'pyCEA_path')
from pyCEA import cea

You can then use 
results = cea(inputs)
isp = cea(inputs)['ISP'][-1]

by Alexis Angers
"""


def cea_float(s):
    if len(s) < 3:
        return float(s)
    elif (s[-2] == '-')|(s[-3] == '-')|(s[-3] == '-'):
        i = len(s)-1
        found = False
        while not(found):
            if s[i] == '-':
                found = True
            else:
                i -= 1
        num1 = float( s[:(i-1)])
        num2 = float( s[i:])
        return num1 * 10**num2
    else:
        if '*' in s:
            return 0
        else:
            return float(s)


def write_cea_input(inpts, inputname='cea_run'):

    # File generation
    thisdir = str(Path(__file__).parent.absolute())
    last_path = str(Path().absolute())
    chdir(thisdir)
    with open(inputname + '.inp', 'w') as f:

        # Print start things
        print('problem', file=f)
        print('rocket \t equilibrium', file=f)
        print('p,psi=' + '{:0.6f}'.format(inpts['P_CC']), file=f)
        print('o/f=' + '{:0.6f}'.format(inpts['OF']), file=f)
        print('pip ' + '{:0.6f}'.format(inpts['P_CC']/inpts['P_EXT']), file=f)
        print('reacts ', file=f)

        # Print each fuel in the fuel dictionary
        for fuel in inpts['fuels']:
            str2add = 'fuel = ' + fuel['name']
            if 'wt' in fuel:
                str2add += '  wt%%=' + '{:.6f}'.format(fuel['wt'])
            if 'T' in fuel:
                str2add += '  t,k=' + '{:.6f}'.format(fuel['T'])
            if 'h' in fuel:
                str2add += '  h,kj/mol=' + '{:.3f}'.format(fuel['h'])
            if 'comp' in fuel:
                str2add += '  ' + fuel['comp']
            print(str2add, file=f)

        for oxid in inpts['oxidizers']:
            str2add = 'oxid = ' + oxid['name']
            if 'wt' in oxid:
                str2add += '  wt%%=' + '{:.6f}'.format(oxid['wt'])
            if 'T' in oxid:
                str2add += '  t,k=' + '{:.6f}'.format(oxid['T'])
            if 'h' in oxid:
                str2add += '\n  h,kj/mol=' + '{:.6f}'.format(oxid['h'])
            if 'comp' in oxid:
                str2add += '  ' + oxid['comp']
            print(str2add, file=f)

        if 'insert' in inpts:
            insert_str = ''
            for ins in inpts['insert']:
                insert_str += ins + ' '
            print('insert '+insert_str, file=f)
        
        if 'output' in inpts:
            print('output ' + inpts['output'], file=f)
        else:
            print('output short', file=f)
        print('end', file=f)
    chdir(last_path)


def run_cea(file_name):
    thisdir = str(Path(__file__).parent.absolute())
    last_path = str(Path().absolute())
    chdir(thisdir)
    with open(file_name + '.txt','w') as f:
        print(file_name, file=f)
    system('FCEA2 < ' + file_name + '.txt > NUL')
    remove(file_name + '.txt')
    remove(file_name + '.inp')
    chdir(last_path)


def read_cea_output(file_name, raw_str=False):
    out = {'run_name':file_name}
    thisdir = str(Path(__file__).parent.absolute())
    last_path = str(Path().absolute())
    chdir(thisdir)
    with open(file_name + '.out','r') as file:
        if raw_str:
            out = file.read()
        else:
            f = file.readlines()
            frac_lines = False
            for line in f:
                if frac_lines:
                    if line.split() == []:
                        frac_tag += 1
                    else:
                        line = line.split()
                        for i in range(len(line)-1):
                            line[i+1] = float(line[i+1])
                        out['mole_frac'][line[0].replace('*','')] = line[1:]
                    if frac_tag > 1:
                        frac_lines = False
                else:
                    if line[:8] == ' Pinf/P ':
                        out['pinf/p'] = [cea_float(s) for s in line[7:].split()]
                    elif line[:5] == ' T, K':
                        out['t'] = [cea_float(s) for s in line[5:].split()]
                    elif line[:13] == ' RHO, KG/CU M':
                        out['rho'] = [cea_float(s) for s in line[13:].split()]
                    elif line[:9] == ' M, (1/n)':
                        out['m'] = [cea_float(s) for s in line[9:].split()]
                    elif line[:15] == ' Cp, KJ/(KG)(K)':
                        out['cp'] = [cea_float(s) for s in line[15:].split()]
                    elif line[:7] == ' GAMMAs':
                        out['gamma'] = [cea_float(s) for s in line[7:].split()]
                    elif line[:14] == ' SON VEL,M/SEC':
                        out['c'] = [cea_float(s) for s in line[14:].split()]
                    elif line[:12] == ' MACH NUMBER':
                        out['mach'] = [cea_float(s) for s in line[12:].split()]
                    elif line[:6] == ' Ae/At':
                        out['ae/at'] = [cea_float(s) for s in line[6:].split()]
                    elif line[:13] == ' CSTAR, M/SEC':
                        out['cstar'] = [cea_float(s) for s in line[13:].split()]
                    elif line[:3] == ' CF':
                        out['cf'] = [cea_float(s) for s in line[3:].split()]
                    elif line[:12] == ' Ivac, M/SEC':
                        out['ivac'] = [cea_float(s) for s in line[12:].split()]
                    elif line[:11] == ' Isp, M/SEC':
                        out['isp'] = [cea_float(s) for s in line[11:].split()]
                    elif line[:15] == ' MOLE FRACTIONS':
                        out['mole_frac'] = {}
                        frac_lines = True
                        frac_tag = 0


    remove(file_name + '.out')
    chdir(last_path)
    return out


def cea(inpts, run_name=[], raw_str=False):
    if run_name == []:
        run_name = randint(1000000,9999999)
        run_name = str(run_name)

    write_cea_input(inpts, run_name)
    run_cea(run_name)
    return read_cea_output(run_name, raw_str)



if __name__=='__main__':
    # Input handling
    hstr = 'pyCEA.py -f <fuelname-fuel wt%-fuel temp.-fuel comp./fuelname2...> -o <oxidname-oxid wt%-oxid temp.-oxid comp./oxidname2...> -p <comb. pressure (psia)> -b <backpressure (psia)> -r <oxid/fuel ratio> -i <species1/species2/... -e <output>'
    try:
        opts, args = getopt.getopt(sys.argv, "f:o:p:b:r:i:", ["fuel=","oxid=","pc=","pe=","of=","insert="])
    except getopt.GetoptError:
        print(hstr)
        sys.exit(2)

    fuels = []
    oxids = []
    insert = []
    output = ''
    Pc = 0
    Pb = 0
    OF = 1

    i = 0
    while i < (len(args)-1)/2:
        i += 1
        opt = args[2*i-1]
        if len(args)>2:
            arg = args[2*i]
        if opt == "-h":
            print(hstr)
            sys.exit()
        elif opt in ("-f","--fuel"):
            fuels_str = arg.split('/')
            for fuel_str_i in fuels_str:
                fuelstr = fuel_str_i.split('-')
                fuel = {'name':fuelstr[0]}
                if len(fuelstr)>1:
                    fuel['wt']=float(fuelstr[1])
                if len(fuelstr)>2:
                    fuel['T']=float(fuelstr[2])
                if len(fuelstr)>3:
                    fuel['comp']=fuelstr[3]
                fuels.append(fuel)
        elif opt in ("-o","--oxid"):
            oxids_str = arg.split('/')
            for oxid_str_i in oxids_str:
                oxidstr = oxid_str_i.split('-')
                oxid = {'name':oxidstr[0]}
                if len(oxidstr)>1:
                    oxid['wt']=float(oxidstr[1])
                if len(oxidstr)>2:
                    oxid['T']=float(oxidstr[2])
                if len(oxidstr)>3:
                    oxid['comp']=oxidstr[3]
                oxids.append(oxid)
        elif opt in ("-p","--pc"):
            Pc = float(arg)
        elif opt in ("-b","--pb"):
            Pb = float(arg)
        elif opt in ("-r","--of"):
            OF = float(arg)
        elif opt in ("-i","--insert"):
            insert_str = arg.split('/')
            insert = insert_str
        elif opt in ("-e","--output"):
            output = arg

    inpts = {
        'P_CC':Pc,
        'P_EXT':Pb,
        'OF':OF,
        'fuels':fuels,
        'oxidizers':oxids,
    }
    if insert != []:
        inpts['insert'] = insert
    if output != '':
        inpts['output'] = output

    print(cea(inpts, raw_str=True))

