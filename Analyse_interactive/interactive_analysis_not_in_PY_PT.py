import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from PIL import ImageTk
from PIL import Image
import os
import math
import subprocess

#relative to mot file, or could be in Anlayse_interactive
path_to_csv = "c:\\Users\\colin\\OneDrive\\Desktop\\Save\\Python Code\\Sim\\Analyse_interactive\\OX_W_ANALYSIS.csv"
separator = ","
indepedent_variable_name = "time"


class dataset:
    def __init__(self, csv_file, separator, indepedent_variable_name):
        current_directory = os.getcwd()
        print(f"Current directoy : {current_directory}")
        upper_directory = os.path.dirname(current_directory)
        print(f"Upper directoy : {upper_directory}")
        mot_folder = upper_directory+'\\PY-PT'+'\\MOT'
        try:
            self.data = pd.read_csv(mot_folder +'\\' + csv_file, sep=separator)
        except:
            self.data = pd.read_csv(csv_file)
        self.datacopy = self.data
        self.df = self.data
        self.time = indepedent_variable_name
        self.sim = False


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
        self.all = self.main

        self.root = tk.Tk()
        self.first = True

        self.root.title("MAIN MENU")
        self.main_menu()
        self.root.mainloop()

    def clear_widgets(self):
        # Iterate over all children of the window and destroy them
        for widget in self.root.winfo_children():
            widget.destroy()

    def main_menu(self):
        # Create a menubar
        self.clear_widgets()
        menubar = tk.Menu(self.root)

        # Create a menu
        menu = tk.Menu(menubar, tearoff=0)

        # Add menu items
        menu.add_command(label="Resize Data", command=self.resize)
        menu.add_command(label="Plot Data", command=self.interactive_plotting)
        menu.add_command(label="Rename columns", command=self.rename)
        menu.add_command(label="Drop columns", command=self.drop_columns)
        menu.add_command(label="Perform moving average", command=self.moving_average)
        menu.add_command(label="Perform integral", command=self.perform_integral)
        menu.add_command(label="Perform derivative", command=self.perform_derivative)
        menu.add_command(label="Perform fft", command=self.perform_fft)
        menu.add_command(label="Add csv", command=self.add_csv)
        menu.add_command(label="Run simulation", command=self.simulate)
        menu.add_command(label="Save Data", command=self.save)

        # Add the menu to the menubar
        menubar.add_cascade(label="Menu", menu=menu)

        

        self.img = ImageTk.PhotoImage(Image.open("symbol_oronos.png"))
        panel = tk.Label(self.root, image=self.img)
        panel.pack(side="bottom", fill="both", expand="yes")

        self.root.config(menu=menubar)
        self.all = pd.concat([dataset.data for dataset in self.datasets.values()])

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

        for col in liste:
            if ('time' not in col) and ('Time' not in col):
                var = tk.BooleanVar()
                checkbox = tk.Checkbutton(self.root, text=col, variable=var)
                self.checkbox_vars[col] = var
                checkboxes.append(checkbox)

        index = 0
        frame = tk.Frame(self.root)
        frame.grid(row=0, column=0)

        for i in range(math.ceil(len(self.main.data.columns) / 2)):
            for j in range(int(len(self.datasets) * 2)):
                if len(checkboxes) > index:
                    checkboxes[index].grid(row=i, column=j, padx=5, pady=5, sticky="w")
                    index += 1

        tk.Frame(self.root).grid(row=3, column=0)

    def finalize_selection(self, liste : list = None):
        if liste == None:
            liste = self.all.columns
            liste = [col for col in self.all.columns if (('time' not in col) and ('Time' not in col))]

        self.selected_columns = [
            col for col in liste if self.checkbox_vars[col].get()
        ]

    def plot_with_m_avg(self, col):
        
        print(f"Plotting change asked")
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
                print(f"Makw change asked")
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
                print("Name of dataset : {}".format(name))
                print("T before return : {}".format(dataset.data[dataset.time]))
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
        plt.legend()
        plt.show()

    def interactive_plotting(self):
        
        self.first = True
        self.clear_widgets()

        if self.first:
            self.select_columns()
            self.first = False

        # Button to plot data
        plot_button = tk.Button(self.root, text="Plot Data", command=self.plot_columns)
        plot_button.grid(row=11, column=0, columnspan=len(self.datasets) * 2, pady=10)

        # Button to go back to the main menu
        button = tk.Button(self.root, text="Main Menu", command=self.main_menu)
        button.grid(row=12, column=0, columnspan=len(self.datasets) * 2, pady=10)

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
            both = self.time_and_df_from_col(value)
            t, dataset = both["time"], both["dataset"]
            print("ITME : ", t)
            value_str = value
            value = dataset.data[value].dropna()
            y = value - value.mean()
            fft_values = np.fft.fft(y)
            print("FFT VALUES : ", fft_values)
            print("MEAN DIF : ", np.mean(np.diff(t.dropna())))

            frequencies = np.fft.fftfreq(len(y), d=np.mean(np.diff(t)))
            print("FRQUENCIES BEFORE TRIM", frequencies)

            amplitudes = np.abs(fft_values)
            positive_freq_idxs = np.where(frequencies > 0)
            print(f"Indexing : ", positive_freq_idxs)
            frequencies = frequencies[positive_freq_idxs]
            print("FREQUENCIES :", frequencies)
            amplitudes = amplitudes[positive_freq_idxs]
            print("AMPLITUDES :", amplitudes)

            plt.plot(frequencies, amplitudes, linewidth=1)
            plt.title("Spectrum des fréquences " + value_str)
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            if ylim != None:
                plt.ylim(0, ylim)
            else:
                amp = amplitudes[frequencies > 25]
                try:
                    plt.ylim(0, max(amp) * 1.1)
                except:
                    print("No amplitude detected, please try with another column")
            if minxlim != None:
                plt.xlim(left=minxlim)
            plt.show()

            amps = amplitudes[frequencies > 25]
            freqs = frequencies[frequencies > 25]
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
                self.place_widget(copy, 6, 10)
                replace = tk.Button(
                    self.root,
                    text="Replace with moving average",
                    command=lambda: replace_mvg_avg(col, active),
                )
                self.place_widget(replace, 7, 10)
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

    def update_all(self):
        self.all = pd.concat(
            [dataset.data for dataset in self.datasets.values()], axis=1
        )


    def add_csv(self):
        label = tk.Label(self.root, text="CSV NAME : ")
        label.pack()
        csv_name = tk.Entry(self.root)
        csv_name.pack()

        label = tk.Label(self.root, text="Separator")
        label.pack()
        separator = tk.Entry(self.root)
        separator.pack()

        label = tk.Label(self.root, text="Independent variable name")
        label.pack()
        indepedent_name = tk.Entry(self.root)
        indepedent_name.pack()

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
        button.pack()

    def simulate(self):
        self.clear_widgets()

        mot_name = 'mot_colin.py'
        current_directory = os.getcwd()
        upper_directory = os.path.dirname(current_directory)
        mot_folder = upper_directory+'//PY-PT'+'//MOT'
        
        files_in_mot = os.listdir(mot_folder)

        mot = os.path.join(upper_directory, 'PY-PT', 'MOT', mot_name)

        jsons = [file for file in files_in_mot if file.endswith('.json')]

        self.select_columns(jsons)

        def run_sim(config, mot = mot_name):

            os.chdir(upper_directory+'//PY-PT'+'//MOT')
            result = subprocess.run(['python', mot, '-f', config],
                        capture_output=True, text=True)
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

            new_csv = dataset(newest_file[0], ',' , 'Time      (s)')
            self.datasets[newest_file[0]] = new_csv
            self.update_all()
            new_csv.sim = True

            self.main_menu()

        def choose_json():
            self.finalize_selection(jsons)
            selected = self.selected_columns[0]
            print(selected)
            
            run_sim_button = tk.Button(self.root, text = 'Run Sim with Choosen config', command = lambda : run_sim(selected))
            self.place_widget(run_sim_button,9,4)
        

        confirm = tk.Button(self.root, text='Confirm config', command = choose_json)
        self.place_widget(confirm,10,4)

    def save(self):
        def save_csv():
            self.all.reset_index(drop=True, inplace=True)
            self.all.to_csv(entry.get() + ".csv")
            self.main_menu()
            
        self.clear_widgets()
        label = tk.Label(self.root, text="Enter the saved name you want")
        label.pack()
        entry = tk.Entry(self.root)
        entry.pack()
        B = tk.Button(self.root, text="Confirm", command=save_csv)
        B.pack()


Interactive_Analysis(csv_file=path_to_csv, sep=separator, time=indepedent_variable_name)
