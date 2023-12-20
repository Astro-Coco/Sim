import matplotlib.pyplot as plt
import csv
import sys
sys.path.insert(1, '..\\')
import utils


def plot_data(data):
    # Plot pressure data
    fig, ax = plt.subplots(2)

    ax[0].set_ylabel('Pressure [PSIG]')
    ax[0].plot(data['time'], data['p_tank'], color='k', label='Tank')
    ax[0].plot(data['time'], data['p_inj'], color='tab:orange', label='Inj.')
    ax[0].plot(data['time'], data['p_comb'], color='tab:red', label='Comb.')
    ax[0].tick_params(direction='in')
    ax[0].legend(loc='upper center',ncol=3)

    color = 'tab:blue'
    ax2 = ax[0].twinx()
    ax2.set_ylabel('NOx mass [kg]')
    ax2.plot(data['time_mass'], data['mass'], color=color)
    ax2.tick_params(direction='in')

    color = 'tab:red'
    ax[1].set_ylabel('Thrust [N]')
    ax[1].set_xlabel('Time [s]')
    ax[1].plot(data['time_thrust'], data['thrust'], color=color)
    ax[1].tick_params(direction='in')

    plt.show()


def movmean(data, N):
    cumsum, moving_aves = [0], []

    for i, x in enumerate(data, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            #can do stuff with moving_ave here
            moving_aves.append(moving_ave)
    
    return moving_aves


def save_data(data):
    print('Saving data...')
    # Save as prop file
    utils.write_json('data/data_filt.json', data)

    # Also save as CSV files
    with open('data/pressure_filt.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["time", "p_tank", "p_inj","p_comb"])
        for i in range(len(data['time'])):
            writer.writerow([ data['time'][i], data['p_tank'][i], data['p_inj'][i], data['p_comb'][i] ])
    
    with open('data/mass_filt.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["time","mass"])
        for i in range(len(data['time_mass'])):
            writer.writerow([ data['time_mass'][i], data['mass'][i] ])

    with open('data/thrust_filt.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["time", "thrust"])
        for i in range(len(data['time_thrust'])):
            writer.writerow([ data['time_thrust'][i], data['thrust'][i] ])

    print('Data saved.')


if __name__=='__main__':
    # Load data
    data = utils.read_json('data\\data_raw.json')

    run = True
    while run:
        plot_data(data)

        do_filt = input('Apply filter (y/n): ')
        if do_filt == 'y':
            print('Length of pressure data: ' + str(len(data['time'])))
            order = int(input('Order of pressure filter: '))
            if order > 1:
                data['p_tank'] = movmean(data['p_tank'], order)
                data['p_inj'] = movmean(data['p_inj'], order)
                data['p_comb'] = movmean(data['p_comb'], order)
                data['time'] = data['time'][0:len(data['p_comb'])]

            print('Length of pressure data: ' + str(len(data['time_mass'])))
            order = int(input('Order of mass filter: '))
            if order > 1:
                data['mass'] = movmean(data['mass'], order)
                data['time_mass'] = data['time_mass'][0:len(data['mass'])]

            print('Length of thrust data: ' + str(len(data['time_thrust'])))
            order = int(input('Order of thrust filter: '))
            if order > 1:
                data['thrust'] = movmean(data['thrust'], order)
                data['time_thrust'] = data['time_thrust'][0:len(data['thrust'])]

        elif do_filt == 'n':
            run = False

    shift = input('Value to shift thrust: ')
    
    if shift == 'auto':
        mval = data['thrust'][0]
        for dat in data['thrust']:
            if dat < mval:
                mval = dat
        shift = mval
        for i in range(len(data['thrust'])):
            data['thrust'][i] += shift
            print('Thrust shifted by '+str(shift))
    else:
        try:
            shift = float(shift)
        except:
            shift = []
        
        if shift != []:
            for i in range(len(data['thrust'])):
                data['thrust'][i] += shift
            print('Thrust shifted by '+str(shift))

    # Save filtered data
    do_save = input('Save data (y/n): ')
    if do_save == 'y':
        save_data(data)

    print('End')
