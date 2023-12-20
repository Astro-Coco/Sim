import matplotlib.pyplot as plt
import sys
import numpy as np
sys.path.insert(1, '..\\')
import utils
from filt_data import plot_data
# Flip two pressure curves in case of bad index guesses


def save_data(data):
    print('Saving data...')
    # Save as prop file
    utils.write_json('data/data_raw.json', data)
    
    print('Data saved.')



if __name__=='__main__':
    # Load data
    data = utils.read_json('data\\data_raw.json')

    run = True
    while run:
        plot_data(data)

        do_flip = input('First pressure to invert (comb/inj/tank/n): ')
        do_flip_2 = input('Second pressure to invert (comb/inj/tank/n): ')
        if (do_flip != 'n')&(do_flip_2 != 'n'):
            temp = data['p_' + do_flip]
            data['p_' + do_flip] = data['p_' + do_flip_2 ]
            data['p_' + do_flip_2 ] = temp
            print(do_flip + ' pressure and ' + do_flip_2 + ' pressure flipped')
        else:
            print('End of flipping procedure.\n')
            run = False

    # Save filtered data
    do_save = input('Save data (y/n): ')
    if do_save == 'y':
        save_data(data)

    print('End')
