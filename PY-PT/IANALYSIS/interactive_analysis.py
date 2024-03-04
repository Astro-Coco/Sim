import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
import customtkinter
from PIL import ImageTk
from PIL import Image
import os
import math
import subprocess

import parameters
import optimisation
import reduce_opti
import gradient_descent

path_to_PYPT = r'C:\Users\Ordinateur\Desktop\Oronos\Sim\PY-PT'
mot_name = 'mot_colin.py'
 
csv_name = 'cleaned_data_02_mars_2024.csv'
separator = ","
indepedent_variable_name = "time"


class dataset:
    def __init__(self, csv_file, separator, indepedent_variable_name):
        file = os.path.join(path_to_PYPT, 'IANALYSIS', 'sauvegardes_csv', csv_file)
        try:
            self.data = pd.read_csv(file, sep=separator)
            print(self.data['time'])
        except:
            file = os.path.join(path_to_PYPT,'MOT', csv_file)
            self.data = pd.read_csv(file, sep=separator)

        self.datacopy = self.data
        self.df = self.data
        self.time = indepedent_variable_name
        self.sim = False
        self.opti = False


class Interactive_Analysis:
    def __init__(self, csv_file: str, time="time", sep=None):
        plt.ion()

        if sep == None:
            sep = ","

            # Add search to figure sep eventually

        self.main = dataset(
            csv_file=csv_file, separator=sep, indepedent_variable_name=time
        )
        self.datasets = {csv_file: self.main}
        self.all = self.main.data


        self.root = tk.Tk()
        self.root.geometry('1300x700')
        self.root.title('Analysis')
        self.root.resizable(width = True,height = True)

        self.first = True

        self.main_menu()
        self.root.mainloop()

    def clear_widgets(self):
        # Iterate over all children of the window and destroy them
        for widget in self.root.winfo_children():
            widget.destroy()
        if not self.menu:    
            button = tk.Button(self.root, text = 'Return to menu', command = self.main_menu)
            self.place_widget(button,20,0)

    def main_menu(self):
        # Create a menubar
        self.style = ttk.Style()
        self.style.configure("Large.TButton", font=("Arial", 12), padding=50)

        # Clear existing widgets
        self.menu = True
        self.clear_widgets()
        self.menu = False
        self.img = ImageTk.PhotoImage(Image.open(os.path.join(path_to_PYPT, 'IANALYSIS', "symbol_oronos.png")))
        panel = tk.Label(self.root, image=self.img)
        panel.grid(row=0, column=0, columnspan=4, padx=20, pady=20)  # Add padding for aesthetics

        # Add buttons
        self.create_button("Plot Data", self.interactive_plotting, 1, 0, "Large.TButton")
        self.create_button("Run simulation", self.simulate, 1, 1, "Large.TButton")
        self.create_button("Rename columns", self.rename, 6, 0)
        self.create_button("Resize Data", self.resize, 2, 0)
        self.create_button("Drop columns", self.drop_columns, 3, 0)
        self.create_button("Drop datasets", self.drop_datasets, 4, 0)
        self.create_button("Add csv", self.add_csv, 5, 0)
        self.create_button("Perform moving average", self.moving_average, 4, 1)
        self.create_button("Perform integral", self.perform_integral, 5, 1)
        self.create_button("Perform derivative", self.perform_derivative, 6, 1)
        self.create_button("Perform fft", self.perform_fft, 3, 1)
        self.create_button("Optimize sim parameter", self.optimize, 2, 1)
        self.create_button("Gradient descent", self.gradient_opti, 7,1)
        self.create_button('Produce reference thrust', self.produce_ref_data, 7, 0)
        self.create_button("Save Data", self.save, 8, 0, columnspan=2)


    def create_button(self, text, command, row, column, style=None, columnspan=1):
        if style:
            button = ttk.Button(text=text, command=command, style=style)
        else:
            button = ttk.Button(text=text, command=command)
        button.grid(row=row, column=column, columnspan=columnspan, padx=10, pady=10, sticky="nsew")



    def place_widget(self, widget, row, column, rowspan=1, columnspan=1):
        widget.grid(
            row=row,
            column=column,
            rowspan=rowspan,
            columnspan=columnspan,
            padx=5,
            pady=5,
            sticky="w",
        )

    def rename(self):
        
        self.clear_widgets()
        self.select_columns()

        def ask_names():
            self.finalize_selection()
            self.clear_widgets()
            intro = tk.Label(self.root, text="Please enter the names you want")
            self.place_widget(intro, 0, 0)

            entries = {}
            column_placement = 3
            threshold = 4
            index = 0
            for column in self.selected_columns:
                if index > threshold:
                    column_placement += 1
                    index = 0

                label = tk.Label(self.root, text=column)
                self.place_widget(label, index * 2, column_placement)
                entry = tk.Entry(self.root)
                self.place_widget(entry, index * 2 + 1, column_placement)
                entries[column] = entry
                index += 1

            def rename_columns():
                for col, entry_widget in entries.items():
                    new_name = entry_widget.get()
                    self.all.rename(columns={col: new_name}, inplace=True)
                    self.time_and_df_from_col(col)["dataset"].data.rename(
                        columns={col: new_name}, inplace=True
                    )

                self.main_menu()

            confirm = tk.Button(
                self.root, text="Confirm changes", command=rename_columns
            )
            self.place_widget(confirm, 1, 0)

        button = tk.Button(self.root, text="Confirm selection", command=ask_names)
        self.place_widget(button, 10, 3)

    def select_columns(self, liste : list = None):
        if liste == None:
            liste = self.all.columns

        self.checkbox_vars = {}
        checkboxes = []


       


        for index, col in enumerate(liste):
            if ('time' not in col) and ('Time' not in col):
                var = tk.BooleanVar()
                checkbox = tk.Checkbutton(self.root, text=col, variable=var)
                
                self.checkbox_vars[col] = var
                checkboxes.append(checkbox)

            #self.current_shape = checkbox[index].winfo.width(), checkbox[index].winfo.height()

        index = 0
        frame = tk.Frame(self.root)
        frame.grid(row=0, column=0)
        n_elements = len(liste)
        screen_width, screen_height = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        aspect_ratio = screen_width / screen_height
        #aspect_ratio = 0.8
        I, J = math.sqrt(aspect_ratio*n_elements), math.sqrt(n_elements/aspect_ratio)
        for j in range(math.floor(J)):
            for i in range(math.ceil(I)):
                if len(checkboxes) > index:
                    checkboxes[index].grid(row=i, column=j, padx=5, pady=5, sticky="w")
                    index += 1

        tk.Frame(self.root).grid(row=3, column=0)

    def finalize_selection(self, liste : list = None):
        if liste == None:
            liste = self.all.columns
        
        liste = [col for col in liste if (('time' not in col) and ('Time' not in col))]

        self.selected_columns = [
            col for col in liste if self.checkbox_vars[col].get()
        ]

    def plot_with_m_avg(self, col):
        
        
        plt.cla()
        dataset = self.time_and_df_from_col(col)["dataset"]
        df = dataset.df
        t = df[dataset.time]
        y = df[col]
        plt.scatter(t, y, s=2)
        plt.scatter(t, y.rolling(window=40).mean(), s=1)
        plt.show()

    def resize(self):
        
        self.clear_widgets()
        self.select_columns()

        def modify(self=self):
            self.clear_widgets()
            self.last_i = self.main.df[self.main.time].min()
            self.last_f = 0
            self.finalize_selection()
            self.col_name = self.selected_columns[0]
            self.plot_with_m_avg(self.col_name)

            entry = tk.Entry(self.root, name="modification")
            self.place_widget(entry, 0, 4)
            self.changing_initial = True

            def switch(self=self):
                if self.changing_initial:
                    self.changing_initial = False
                    choose.config(text="Final change")
                else:
                    self.changing_initial = True
                    choose.config(text="Initial change")

            def make_change(self=self):
                
                both_info = self.time_and_df_from_col(self.col_name)
                time, dataset = both_info["time"], both_info["dataset"]
                data = dataset.data
                entree = float(entry.get())
                if self.changing_initial:
                    self.last_i += entree
                else:
                    self.last_f += entree

                dataset.df = data[
                    (time < (time.max() + self.last_f))
                    & (time > (time.min() + self.last_i))
                ]

                dataset.df[dataset.time] = (
                    dataset.df[dataset.time] - dataset.df[dataset.time].min()
                )
                self.plot_with_m_avg(col=self.col_name)

            def end():
                dataset = self.time_and_df_from_col(self.col_name)["dataset"]
                dataset.data = dataset.df
                self.update_all()
                self.main_menu()

            choose = tk.Button(self.root, text="Initial change", command=switch)
            self.place_widget(choose, 1, 4)
            confirm = tk.Button(
                self.root, text="Plot change", command=make_change, pady=20
            )
            self.place_widget(confirm, 3, 4)
            menu = tk.Button(self.root, text="Save to Menu", command=end, pady=20)
            self.place_widget(menu, 8, 8)

        choose_column = tk.Button(
            self.root, text="Confirm column choice", command=modify, pady=20, padx=20
        )
        self.place_widget(choose_column, 10, 10)

    def time_and_df_from_col(self, col):
        
        print("COLUMN NAME ASKED : ", col)
        for name, dataset in self.datasets.items():
            if col in dataset.data.columns:
                return {"time": dataset.data[dataset.time], "dataset": dataset}
        else:
            print("No corresponding df found")

    def plot_columns(self):
        
        self.finalize_selection()
        plt.clf()

        
        
        for col in self.selected_columns:
            print(self.time_and_df_from_col(col)["time"])
            plt.scatter(
                x=self.time_and_df_from_col(col)["time"],
                y=self.time_and_df_from_col(col)["dataset"].data[col],
                s=2,
                label=col,
            )
        plt.grid(visible = True, which = "both")
        plt.minorticks_on()
        plt.legend(loc = 'best', fontsize = 'small')
        plt.show()

    def interactive_plotting(self):
        
        self.first = True
        self.clear_widgets()

        if self.first:

            opti_liste = []
            for datasete in self.datasets.values():
                columns = datasete.data.columns
                if datasete.opti:
                    opti_liste.extend(columns)

            opti_liste = sorted(opti_liste)

            liste = [col for col in self.all.columns if col not in opti_liste]
            liste.extend(opti_liste)
 
            self.select_columns(liste)
            self.first = False

        # Button to plot data
        plot_button = tk.Button(self.root, text="Plot Data", command=self.plot_columns)
        plot_button.grid(row=20, column=3, columnspan=len(self.datasets) * 2, pady=10)

        # Button to go back to the main menu
        button = tk.Button(self.root, text="Main Menu", command=self.main_menu)
        button.grid(row=21, column=3, columnspan=len(self.datasets) * 2, pady=10)

    def perform_derivative(self):
        
        self.clear_widgets()
        self.select_columns()

        def derivative(self=self):
            plt.cla()
            self.finalize_selection()
            y_name = self.selected_columns[0]
            dataset = self.time_and_df_from_col(y_name)["dataset"]

            x = dataset.data[dataset.time]

            # Calculate the derivative
            dy_dx = np.gradient(dataset.data[y_name], x)
            mean_dx = pd.Series(dy_dx).rolling(window=30).mean()
            mini = mean_dx[10:].min()
            maxi = mean_dx[10:].max()

            plt.scatter(x, mean_dx, s=2)
            plt.title(f"Derivative of {y_name}")
            plt.text(0.3, 0.2, f"Max derivative value: {max(mean_dx):.2f}")
            plt.xlabel("Time (s)")
            plt.ylabel("Derivative")
            plt.ylim(mini - abs(mini) * 0.1, maxi * 1.1)
            plt.show()

            # Store the derivative in the data

            dataset.data[y_name + "_derivative"] = dy_dx
            self.update_all()
            self.main_menu()

        button = tk.Button(
            self.root, text="Plot Derivative", command=derivative, padx=10, pady=5
        )
        self.place_widget(button, 10, 10)

    def perform_integral(self):
        
        self.clear_widgets()
        self.select_columns()

        def integral(self=self):
            plt.cla()
            self.finalize_selection()
            y_name = self.selected_columns[0]
            dataset = self.time_and_df_from_col(y_name)["dataset"]

            x = dataset.data[dataset.time]
            integral = np.array(
                [
                    np.trapz(dataset.data[y_name][: i + 1], x[: i + 1])
                    for i in range(len(dataset.data[y_name]))
                ]
            )
            plt.plot(x, integral)
            plt.title(f"Integral of {y_name}")
            plt.text(0.3, 0.2, f"Integral value : {max(integral):.2f}")
            plt.xlabel("Time (s)")
            plt.show()

            dataset.data[y_name + "_integral"] = integral
            self.update_all()
            self.main_menu()

        button = tk.Button(self.root, text="Plot", command=integral, padx=10, pady=5)
        self.place_widget(button, 10, 10)

    def perform_fft(self):
        
        self.clear_widgets()
        self.first = True
        label = tk.Label(
            self.root,
            text=f"Check 1 column to perform fourrier transform",
        )
        self.place_widget(label, 8, 10)

        self.select_columns()

        def fft(ylim=None, minxlim=10):
            plt.cla()
            self.finalize_selection()
            value = self.selected_columns[0]
            print('column chose : ' ,value)
            both = self.time_and_df_from_col(value)
            t, dataset = both["time"], both["dataset"]
            print('time : ', t)

            value_str = value
            value = dataset.data[value].dropna()
            y = value - value.mean()
            print('y : ', y)
 
            fft_values = np.fft.fft(y)
            print('fft_values', fft_values)
        
            frequencies = np.fft.fftfreq(len(y), d=np.mean(np.diff(t[:len(y)])))
            print('len_y : ', len(y))
            print('freqs from fft : ' , frequencies)

            amplitudes = np.abs(fft_values)
            print(amplitudes)
            positive_freq_idxs = np.where(frequencies > 0)
            
            frequencies = frequencies[positive_freq_idxs]
            print('freqs1  : ' , amplitudes)
            amplitudes = amplitudes[positive_freq_idxs]
            print('amps1  : ' , amplitudes)
            

            plt.plot(frequencies, amplitudes, linewidth=1)
            plt.title("Spectrum des fréquences " + value_str)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            if ylim != None:
                plt.ylim(0, ylim)
            else:
                #amp = amplitudes[frequencies > 25]
                pass
                try:
                   # plt.ylim(0, max(amp) * 1.1)
                    pass
                except:
                    print("No amplitude detected, please try with another column")
            if minxlim != None:
                plt.xlim(left=minxlim)
            plt.show()

            amps = amplitudes[frequencies > 25]
            print(f'amplitudes : ', amps)
            freqs = frequencies[frequencies > 25]
            print(f'frequencies : ', freqs)
            print(f"max resonnance : {freqs[amps == max(amps)][0]}")

            self.main_menu()
            return frequencies, amplitudes

        button = tk.Button(self.root, text="Plot", command=fft)
        self.place_widget(button, 10, 10)

    def moving_average(self):
        
        self.clear_widgets()
        self.first = True

        self.select_columns()

        def replace_mvg_avg(col, active):
            data = self.time_and_df_from_col(col)["dataset"].data
            data[col] = active
            self.update_all()
            self.main_menu()

        def copy_mvg_avg(col, active):
            data = self.time_and_df_from_col(col)["dataset"].data
            data[col + "_m_avg"] = active
            self.update_all()
            self.main_menu()

        def plot_mvg():
            plt.cla()
            self.finalize_selection()
            col = self.selected_columns[0]
            both = self.time_and_df_from_col(col)
            time, dataset = both["time"], both["dataset"]
            data = dataset.data
            active = data[col].rolling(window=slider.get(), min_periods=1).mean()
            plt.scatter(time, data[col], s=2)
            plt.scatter(time, active, s=2)
            plt.show()

            if self.first:
                copy = tk.Button(
                    self.root,
                    text="Save moving average",
                    command=lambda: copy_mvg_avg(col, active),
                )
                self.place_widget(copy, 8, 10)
                replace = tk.Button(
                    self.root,
                    text="Replace with moving average",
                    command=lambda: replace_mvg_avg(col, active),
                )
                self.place_widget(replace, 9, 10)
                self.first = False

        slider = tk.Scale(self.root, from_=1, to=100, orient="horizontal")
        self.place_widget(slider, 10, 2)

        button = tk.Button(self.root, text="Plot", command=lambda: plot_mvg())
        self.place_widget(button, 11, 2)

    def drop_columns(self):
        self.clear_widgets()

        def keep():
            self.finalize_selection()
            for dataset in self.datasets.values():
                print('SELECTED', self.selected_columns)
                print('data.columns : ', dataset.data.columns)
                cols = [col for col in dataset.data.columns if col in self.selected_columns]
                dataset.data.drop(labels= cols, axis = 1, inplace = True)
                self.update_all()

            
            self.main_menu()

        label = tk.Label(self.root, text="Please check columns you want to drop")
        self.place_widget(label, 10, 5)

        self.select_columns()

        button = tk.Button(self.root, text="Confirm choices", command=keep)
        self.place_widget(button, 10, 6)

    def drop_datasets(self):
        self.clear_widgets()
        keys = self.datasets.keys()
        self.select_columns(keys)

        def drop_those():
            self.finalize_selection(keys)

            for key in self.selected_columns:
                del self.datasets[key]
            
            self.main_menu()
            self.update_all()

        button = tk.Button(self.root, text = 'Drop datasets', command = drop_those)
        self.place_widget(button,10,10)


    def update_all(self):

        if len(self.datasets) != 0:
            self.all = pd.concat(
                [dataset.data.reset_index(drop = True) for dataset in self.datasets.values()], axis=1
            )


    def add_csv(self):
        self.clear_widgets()
        label = tk.Label(self.root, text="CSV NAME : ")
        self.place_widget(label,10,9)
        csv_name = tk.Entry(self.root)
        self.place_widget(csv_name,10,10)

        label = tk.Label(self.root, text="Separator")
        self.place_widget(label,10,11)
        separator = tk.Entry(self.root)
        self.place_widget(separator,10,12)

        label = tk.Label(self.root, text="Independent variable name")
        self.place_widget(label,10,13)
        indepedent_name = tk.Entry(self.root)
        self.place_widget(indepedent_name,10,14)

        def get_entries():
            name = csv_name.get()
            sepa = separator.get()
            ind_name = indepedent_name.get()
            new_csv = dataset(name, sepa, ind_name)
            if 'Time      (s)' in new_csv.data.columns:
                new_csv.sim = True

            self.datasets[name] = new_csv
            self.update_all()
            self.main_menu()

        button = tk.Button(self.root, text="Confirm Entries", command=get_entries)
        self.place_widget(button,9,10)

    def simulate(self):
        self.clear_widgets()

        current_directory = os.getcwd()

        mot_folder = os.path.join(path_to_PYPT, 'MOT')
        files_in_mot = os.listdir(mot_folder)

        jsons = [file for file in files_in_mot if file.endswith('.json')]

        self.select_columns(jsons)

        def run_sim(config, mot = mot_name):

            os.chdir(os.path.join(path_to_PYPT, 'MOT'))
            print('Config passed : '  , config)
            print('mot name ', mot)
            result = subprocess.run(['python', mot, '-f', config],
                        capture_output=True, text=True)
            print('OUTPUT : ' , result.stdout, ' : OVER')
            os.chdir(current_directory)

            file_times = [(file, os.path.getmtime(os.path.join(mot_folder, file))) for file in files_in_mot if os.path.isfile(os.path.join(mot_folder, file))]

            # Find the file with the latest modification time
            newest_file = max(file_times, key=lambda x: x[1], default=None)
            print(newest_file)

            for name, datasete in self.datasets.items():
                if datasete.sim:
                    del self.datasets[name]
                    self.update_all()
                    break
            print('csv name' , newest_file[0])
            new_csv = dataset(newest_file[0], ',' , 'Time      (s)')
            self.datasets[newest_file[0]] = new_csv
            self.update_all()
            new_csv.sim = True

            self.main_menu()

        def choose_json():
            self.finalize_selection(jsons)
            selected = self.selected_columns[0]
            print('config selected ', selected)
            run_sim_button = tk.Button(self.root, text = 'Run Sim with Choosen config', command = lambda : run_sim(selected))
            self.place_widget(run_sim_button,9,4)
        

        confirm = tk.Button(self.root, text='Confirm config', command = choose_json)
        self.place_widget(confirm,10,4)

    def optimize(self):
        self.clear_widgets()
        mot_folder = os.path.join(path_to_PYPT, 'MOT')
        files_in_mot = os.listdir(mot_folder)

        jsons = [file for file in files_in_mot if file.endswith('.json')]

        self.select_columns(liste = jsons)

        def modify_config():
            self.finalize_selection(liste = jsons)
            
            chosen = self.selected_columns[0]
            values = parameters.get_values_and_paths(os.path.join(path_to_PYPT,'MOT',chosen))
            

            self.clear_widgets()

            param = [f'{value[1]} : {value[0]}' for value in values]

            self.select_columns(param)

            def choose_param():
                self.finalize_selection(liste = param)
                print(self.selected_columns[0])

                self.clear_widgets()

                text = tk.Label(self.root, text = self.selected_columns[0])
                self.place_widget(text,1,1)

                entry1 = tk.Entry(self.root)
                label1 = tk.Label(self.root, text = 'Upper bound')
                entry2 = tk.Entry(self.root)
                label2 = tk.Label(self.root, text = 'Lower bound')

                self.place_widget(label1,2,4)
                self.place_widget(entry1,3,4)
                self.place_widget(label2,2,3)
                self.place_widget(entry2,3,3)

                def confirm_range_and_run():
                    min_range = float(entry2.get())
                    max_range = float(entry1.get())
                    direction_to_param = self.selected_columns[0].split(':')[0].strip().split('_')

                    optimisation.optimise(max_range,min_range,direction_to_param,path_to_PYPT, chosen, mot_name)


                    self.clear_widgets()

                    choices = ['Time      (s)', 'ISP     (m/s)', 'C*      (m/s)', 'Pe/Pb     (-)','Thrust    (N)', 'Impulse  (Ns)', 'P tank (psia)', 'P inj. (psia)','P comb (psia)', 'P crit (psia)', 'O/F       (-)', 'm. ox. (kg/s)','Gox (kg/s-m2)', 'r.     (mm/s)', 'Ullage       ']
                    self.select_columns(choices)

                    def finalize():
                        self.finalize_selection(choices)
                        to_import = reduce_opti.extract(path_to_PYPT, selected=self.selected_columns)

                        keys_to_delete = []

                        for dataset_name, datasete in self.datasets.items():
                            if datasete.opti:
                                keys_to_delete.append(dataset_name)

                        # Delete items outside the loop
                        for key in keys_to_delete:
                            del self.datasets[key]


                        for key, value in to_import.items():
                            one_modification = dataset(value[0],',', value[1])
                            one_modification.opti = True
                            self.datasets[key] = one_modification
                        
                        for key,value in self.datasets.items():
                            if 'Unnamed: 0' in value.data.columns:
                                value.data.drop(['Unnamed: 0'],axis = 1,inplace = True)

                        self.update_all()
                        self.main_menu()

                    
                    finaliser = tk.Button(self.root, text = 'Choose columns to keep',command = finalize)
                    self.place_widget(finaliser,10,10)
                    
                confirm_range = tk.Button(self.root, text = 'Confirm Range', command = confirm_range_and_run)
                self.place_widget(confirm_range,3,5)

            choose_param_button = tk.Button(self.root, text = 'Choose Parameter', command = choose_param)
            self.place_widget(choose_param_button,10,10)

        button = tk.Button(self.root, text = 'Modify choosen config', command = modify_config)
        self.place_widget(button,10,10)

    def produce_ref_data(self):
        self.clear_widgets()

        label = tk.Label(self.root, text= 'Choose reference thrust data to prepare gradient descent\nThe data should be trimmed to high thrust, steady state phase')
        self.place_widget(label,20,0)
        self.select_columns()
        def choose_thrust():
            self.finalize_selection()
            column = self.selected_columns[0]
            print(column)
            saving = os.path.join(path_to_PYPT, 'IANALYSIS', 'descent', 'ref.csv')
            
            all = self.all[[self.time_and_df_from_col(column)['dataset'].time, column]].rename(columns = {self.time_and_df_from_col(column)['dataset'].time: 'time', column : 'ref_thrust'})
            all.to_csv(saving)
            self.main_menu()

        choose_thrust = tk.Button(self.root, text = 'Confirm choice', command = choose_thrust)
        self.place_widget(choose_thrust,15,1)


    def gradient_opti(self):
        self.clear_widgets()
        mot_folder = os.path.join(path_to_PYPT, 'MOT')
        files_in_mot = os.listdir(mot_folder)

        jsons = [file for file in files_in_mot if file.endswith('.json')]

        self.select_columns(liste = jsons)

        def choosen_config():
            self.finalize_selection(liste = jsons)
            path = os.path.join(mot_folder,self.selected_columns[0])

            gradient_descent.gradient_descent(path_to_PYPT).main(path)

        chose_config = tk.Button(self.root, text = 'Choose config', command = choosen_config)
        self.place_widget(chose_config, 10,0)

    def save(self):
        self.clear_widgets()
        def save_csv():
            self.all.reset_index(drop=True, inplace=True)
            self.all.to_csv(os.path.join(path_to_PYPT,'IANALYSIS','sauvegardes_csv',(entry.get() + ".csv")))
            self.main_menu()

        label = tk.Label(self.root, text="Enter the saved name you want (no .csv needed)")
        self.place_widget(label, 1,1)
        entry = tk.Entry(self.root)
        self.place_widget(entry,1,2)
        B = tk.Button(self.root, text="Confirm", command=save_csv)
        self.place_widget(B,1,3)

if __name__ == "__main__":
    Interactive_Analysis(csv_file=csv_name, sep=separator, time=indepedent_variable_name)
