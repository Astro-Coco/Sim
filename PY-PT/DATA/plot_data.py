import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.insert(1, '..\\')
import utils


def plot_data(data, ptag='-'):
    # Plot pressure data
    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 4)

    ax.set_ylabel('Pressure (PSIG)')
    ax.set_xlabel('Time (s)')
    x_time = np.array(data['time']) - data['time'][0]
    ax.plot(x_time, data['p_tank'], ptag, color='k', linewidth=1 , label='Tank')
    ax.plot(x_time, data['p_inj'], ptag, color='tab:orange', linewidth=1 , label='Injection')
    ax.plot(x_time, data['p_comb'], ptag, color='tab:red', linewidth=1 , label='Combustion')
    ax.tick_params(direction='in')
    ax.legend()#bbox_to_anchor=(0.5,1.02), loc="lower center",ncol=3)
    #plt.grid()
    plt.savefig('fig-pressure.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 4)
    color = 'k'#'tab:blue'
    ax.set_ylabel('NOx mass (kg)')
    ax.set_xlabel('Time (s)')
    x_time = np.array(data['time_mass']) - data['time_mass'][0]
    ax.plot(x_time, data['mass'], color=color, linewidth=1)
    ax.tick_params(direction='in')
    #plt.grid()
    plt.savefig('fig-mass.png')

    fig, ax = plt.subplots()
    fig.set_size_inches(4.5, 4)
    color = 'k'#'tab:red'
    ax.set_ylabel('Thrust (kN)')
    ax.set_xlabel('Time (s)')
    x_time = np.array(data['time_thrust']) - data['time_thrust'][0] - 0.5
    ax.plot(x_time, np.array(data['thrust'])/1e3, color=color, linewidth=1)
    ax.tick_params(direction='in')
    #plt.grid()
    plt.savefig('fig-thrust.png')

    plt.show()


if __name__=='__main__':
    choose_data = input('Choose data to plot (raw/filt): ')
    # Load data
    data = []
    if choose_data == 'filt':
        data = utils.read_json('data/data_filt.json')
        ptag = '-'
    elif choose_data == 'raw':
        data = utils.read_json('data/data_raw.json')
        ptag = '-'

    if data != []:
        plot_data(data, ptag)



