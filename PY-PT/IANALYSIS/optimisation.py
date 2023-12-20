### OPTIMISATION SCRIPT ###

import subprocess
import json
import os
import numpy as np

'''from interactive_analysis import path_to_PYPT
from interactive_analysis import mot_name'''


def optimise(maximum, minimum, direction, path_to_PYPT,config_name, mot_name):
    path_to_config = os.path.join(path_to_PYPT, 'MOT', config_name)
    opti_folder =  os.path.join(path_to_PYPT,'MOT','opti_collect')
    all_files = os.listdir(opti_folder)

    for filename in all_files:
        file_path = os.path.join(opti_folder, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    for scaling in np.linspace(minimum,maximum, 11):
        print(scaling)
        with open(path_to_config, 'r') as file:
            data = json.load(file)
        present = data  
        print('DIRECTION')
        print(direction)  
        for key in direction:
                print(key)

                if key == direction[-1]:
                    present[key] = scaling
                else:
                    present = present[key]

        modified_json_file = f"{config_name[:-5]}_{direction[-1]}_{scaling:.3e}".replace('.', ',') + '.json'
        


        with open(os.path.join(opti_folder, modified_json_file), 'w') as file:

            json.dump(data, file)

        # Run the simulation
        mot_path = os.path.join(path_to_PYPT, 'MOT', mot_name)
        
        result = subprocess.run(['python', mot_path, '-f', modified_json_file, '-s', opti_folder],
                        capture_output=True, text=True)

        # Print the output and error (if any)
        print(result.stdout)
        if result.stderr:
            print("Error:", result.stderr)
