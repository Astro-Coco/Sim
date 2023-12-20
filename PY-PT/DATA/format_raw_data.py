import csv
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.insert(1, '..\\')
import utils
# Format raw data

def import_data():

    # //// Pressure collumn indexes ////
    p_tank = 12
    p_comb = 14
    p_inj = 13
    mass = 15

    # Initialize data storage
    data = {'time':[], 'p_tank':[], 'p_inj':[], 'p_comb':[], 'time_mass':[], 'mass':[], 'time_thrust':[], 'thrust':[]}

    # Read pressure data
    num_lines = sum(1 for line in open('data/pressure.csv'))

    # Initialize progress bar
    print('\nReading pressure data')
    pb = utils.progressBar()
    with open('data/pressure.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=';')
        line_count = 0
        goal = 1000
        for row in csv_reader:
            if line_count > 25:
                data['time'].append(float(row[0]))
                data['time_mass'].append(float(row[0]))
                data['p_tank'].append(float(row[p_tank]))
                data['p_comb'].append(float(row[p_comb]))
                data['p_inj'].append(float(row[p_inj]))
                data['mass'].append(float(row[mass]))
            
            line_count += 1
            if line_count / goal > 1:
                pb.set_progress(line_count/num_lines)
                goal += 1000
    pb.set_progress(1)
    pb.end()
    
    # Read thrust data
    num_lines = sum(1 for line in open('data/thrust.csv'))
    print('\nReading thrust data')
    pb = utils.progressBar()
    with open('data/thrust.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 0
        goal = 1000
        for row in csv_reader:
            if line_count > 22:
                data['time_thrust'].append(float(row[0].replace(',','.')))
                data['thrust'].append(float(row[1].replace(',','.')))
            
            line_count += 1
            if line_count / goal > 1:
                pb.set_progress(line_count/num_lines)
                goal += 1000
    pb.set_progress(1)
    pb.end()

    return data


def crop_data(data):

    #  Crop thrust data
    run = True
    while run:
        plt.plot(data['thrust'])

        x = plt.ginput(-1)
        plt.show()
        
        if len(x) == 2:
            #inds = [ int(x[0][0]) , int(x[1][0]) ]
            inds = [ int(x[0][0]) ] #int(x[1][0]) ]
            
            #data['time_thrust'] = data['time_thrust'][ inds[0]:inds[1] ]
            #data['thrust'] = data['thrust'][ inds[0]:inds[1] ]
            data['time_thrust'] = data['time_thrust'][ inds[0]: ]
            data['thrust'] = data['thrust'][ inds[0]: ]
        else:
            run = False
    
    # Crop pressure data
    run = True
    while run:
        plt.plot(data['p_tank'])
        plt.plot(data['p_inj'])
        plt.plot(data['p_comb'])
        plt.plot(data['mass'])
        
        x = plt.ginput(-1)
        plt.show()
        
        if len(x) == 2:
            inds = [ int(x[0][0]) , int(x[1][0]) ]
            data['time'] = data['time'][ inds[0]:inds[1] ]
            data['p_tank'] = data['p_tank'][ inds[0]:inds[1] ]
            data['p_inj'] = data['p_inj'][ inds[0]:inds[1] ]
            data['p_comb'] = data['p_comb'][ inds[0]:inds[1] ]
            data['time_mass'] = data['time_mass'][ inds[0]:inds[1] ]
            data['mass'] = data['mass'][ inds[0]:inds[1] ]
        else:
            run = False
    
    return data



def save_data(data):
    print('Saving data...')
    utils.write_json('data\\data_raw.json', data)
    print('Data saved')

if __name__=='__main__':
    data = import_data()
    data = crop_data(data)
    save_data(data)
    
