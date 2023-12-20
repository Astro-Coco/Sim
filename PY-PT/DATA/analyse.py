import sys
sys.path.insert(1, '..\\')
sys.path.insert(1, '..\\CEA')
import utils
from pyCEA import cea
# Analyse motor test data

def trapz(x,y):
    I = 0

    for i in range(len(x)-1):
        dx = x[i+1]-x[i]
        I += (y[i]+y[i+1])*dx/2

    return I


def analyse_test(data):
    gdat = utils.read_json('data/global.json')
    I = trapz(data['time_thrust'], data['thrust'])
    print('Impulse ----- : ', I)
    ri = gdat['d0']/2000
    re = gdat['d1']/2000
    rm = (ri+re)/2
    t = gdat['t']
    r_dot = (re - ri)/t # m/s
    m_dot_fuel = gdat['mf']/t
    m_dot_ox = gdat['mo']/t #(22.2-19.6)/t
    m_dot_tot = m_dot_fuel + m_dot_ox
    OF = m_dot_ox/m_dot_fuel
    print('r dot ------- : ', r_dot*1000)
    print('O / F ------- : ', OF)
    Gox = m_dot_ox/(3.1415927*rm**2)
    print('Mean m_dot_ox : ',m_dot_ox)
    print('Mean Gox ---- : ',Gox)
    print('a (n=0.5) --- : ',r_dot/(Gox**0.5))
    
    # Compute mean combustion pressure
    pm = 0
    lenpm = 1
    for p in data['p_comb']:
        if p > 50:
            pm += p
            lenpm += 1
    pm = pm / lenpm
    pm += 14.5
    print('Mean comb. P. : ', pm)
    At = 3.1415927*(gdat['dt']/2000)**2
    CS = 6894.76*pm*At/m_dot_tot
    print('Mean C* ----- : ',CS)

    fuels = [{'name':'paraffin', 'T':293, 'comp':'C 32 H 66'}]
    oxids = [{'name':'N2O', 'T':293}]
    inpts = {
            'P_CC':pm,
            'P_EXT':14.5,
            'OF':OF,
            'fuels':fuels,
            'oxidizers':oxids
            }
    res = cea(inpts)
    CS_theo = res['cstar'][-1]
    ISP_theo = res['isp'][-1]
    print('C* efficiency : ',CS/CS_theo)
    print('Predi. thrust : ', At*pm*6894.76)
    print('Predicted ISP : ', At*pm*6894.76/m_dot_tot)
    print('ISP efficien. : ',(At*pm*6894.76/m_dot_tot)/ISP_theo)
    






if __name__=='__main__':
    data = utils.read_json('data/data_filt.json')
    
    analyse_test(data)
    
