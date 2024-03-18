import numpy as np
import scipy
import pandas as pd
import math
import matplotlib.pyplot as plt
import subprocess
import os
import json

import parameters

class iteration:
    def __init__(self, config_path):
        #self.PY_PATH = gradient_descent().PY_PT_path
        self.current_config = config_path
        self.gradient = []


        current_mse = gradient_descent().mse()


class gradient_descent:
    def __init__(self, PY_PATH = r"c:\Users\colin\SIM_REPO\Sim\PY-PT"):
    
        self.PY_PT_path = PY_PATH
        self.descent_folder = os.path.join(self.PY_PT_path, "IANALYSIS", "descent")
        
        self.iterations = 0

    def mse(burn_time, ref_time, thrust_to_interp, thrust) -> float:

        interpolated_thrust = np.interp(burn_time, ref_time, thrust_to_interp, right = 0)
        mse = np.mean((interpolated_thrust - thrust) ** 2)

        return mse

    def modify_config(self,gradient, config_path, direction, rate):
        if gradient:
             
            with open(config_path, 'r') as file:
                data = json.load(file)
                current = data

            for key in direction:
                print(key)

                if key == direction[-1]:
                    current[key] *= rate
                else:
                    current = current[key]

            with open(os.path.join(self.descent_folder,f'current_iteration{self.iterations}.json'), 'w') as file:
                json.dump(data, file)

        else:
    
            self.iterations += 1
            
            with open(config_path, 'r') as file:
                data = json.load(file)
                current = data

            for key in direction:
                    print(key)

                    if key == direction[-1]:
                        current[key] *= rate
                    else:
                        current = current[key]

            with open(os.path.join(self.descent_folder,f'current_iteration{self.iterations}.json'), 'w') as file:
                json.dump(data, file)
        

    def compute_gradient(self,params, modified_json_file) -> tuple:
        gradient = []

        rate = 1.01
        # for each parameter, compute partial derivative using param*1.01 (voir la constante de changement)
        for value in params:

            print(value)
            direction = value[2]
            config_path = self.get_last_file()
            self.modify_config(config_path, direction, rate)

        # Need to create new config
        # Run simulation at relatively high dt
        # store the value in a tuple
        pass


    def run_sim(self,mot_name = 'mot_colin.py'):

        # Run the simulation
        mot_path = os.path.join(self,self.PY_PT_path, 'MOT', mot_name)
        config_path = self.get_last_file(extension='.json')
        print(f'DESCENT FOLDER : {self.descent_folder}')

        result = subprocess.run(['python', mot_path, '-f', config_path, '-s', self.descent_folder],
                        capture_output=True, text=True)
        
        print(result.stdout)

    def get_last_file(self, extension=".json"):

        # List all files in the folder
        files = [
            os.path.join(self.descent_folder, f)
            for f in os.listdir(self.descent_folder)
            if os.path.isfile(os.path.join(self.descent_folder, f)) and f.endswith(extension)
        ]

        # Get the most recently updated file
        most_recent_file = max(files, key=os.path.getmtime)

        return most_recent_file

    def confirm_mse(threshold, mse) -> bool:
        if mse < threshold:
            return True
        else:
            return False


    def descent() -> bool:
        # TODO
        # from gradient, choose learnign rate (see how to choose it)
        # Modify new values for the config
        # run simulation at higher dt than compute_gradient()
        # evaluate mse
        # Confirm mse
        # stop if N_th interation achieved or confirme mse = True
        # Save new_config_values
        # plot
        pass


    def main(self, config_path) -> dict:
        descent_folder = os.path.join(self.PY_PT_path, 'IANALYSIS', 'descent')
        
        finished = False
        # mse() #with actual
        values = parameters.get_values_and_paths(config_path)

        allowed_names = [
            "injector_cd",
            "chamber_efficiency",
            "chamber_fuel_a",
            "chamber_fuel_n",
        ]

        allowed_values = [value for value in values if value[1] in allowed_names]
        print(allowed_values)
        #initial_gradient = self.compute_gradient(allowed_values, self.get_last_file())

        with open(config_path, 'r') as file:
            data = json.load(file)

        with open(os.path.join(descent_folder,'main_config.json'), 'w') as file:
            json.dump(data, file)

        self.run_sim()

        while True:
            
            # descent()

            if finished:
                break
            else:
                break


if __name__ == "__main__":
    gradient_descent
