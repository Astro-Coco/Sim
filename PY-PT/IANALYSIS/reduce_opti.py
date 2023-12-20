import os
import pandas as pd
import numpy as np
def extract(path_to_PYPT = 'c:/Users/Ordinateur/Desktop/Oronos/Sim/PY-PT', selected = ['ISP     (m/s)', 'Thrust    (N)']):
    selected.append('Time      (s)')
    print(selected)
    mot_folder = os.path.join(path_to_PYPT, 'MOT', 'opti_collect')
    files_in_mot = os.listdir(mot_folder)

    csvs = [file for file in files_in_mot if file.endswith('.csv')]
    to_import = {}
    for csv in csvs:
        name = csv
        tag = csv.split('_')
        tag = ' ' + tag[-2] + '_' + tag[-1][:-4]
        print(tag)

        path = os.path.join(path_to_PYPT, 'MOT', 'opti_collect', name)
        data = pd.read_csv(path)
        
        columns_to_drop = [col for col in data.columns if col not in selected]
        data.drop(columns_to_drop,axis = 1, inplace = True)
        data.columns = [(col + tag) for col in data.columns]
        
        folder_path = path = os.path.join(path_to_PYPT, 'MOT', 'opti_collect')
        full_path = os.path.join(folder_path, name)
        data.to_csv(full_path)

        time_name = 'Time      (s)' + tag
        to_import[name] = [full_path, time_name]
    else:
        print(to_import)
        return to_import


if __name__ == '__main__':
    extract()