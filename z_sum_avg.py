# ---------------------------- imports and settings ------------------
import os 
import numpy as np


# ---------------------------- class definition and methods ------------------
"""
METHODS
method1:
access to Z1+summary.dat in parallel simulations
extract data columns and compute average
generate Z1+summary_avg.txt

BASIC DATA STRUCTURES
initial Z1+summary.dat 1
    [col1, col2, ..., coln]
...
initial Z1+summary.dat 10
    [col1, col2, ..., coln]

average Z1+summary.dat
    [col1, col2, ..., coln]
"""
class Z1SumAvg:
    """access individual Z1+summary.dat file and compute average for temp/erate"""
    def __init__(self, data_path, n_para, output_path):
        self.data_path = data_path
        self.n_para = n_para
        self.output_path = output_path

    def compute_z1_sum_avg(self):
        """extract data columns, compute average and write avg file"""
        all_data = []
        for i in range(1, self.n_para+1):
            z1_path = f"{self.data_path}/p{i}/Z1+summary.dat"
            if not os.path.exists(z1_path):
                print(f"Missing {z1_path}")
                continue
            z1_data = np.loadtxt(z1_path)
            all_data.append(z1_data)
            if not all_data:
                print(f"There is no Z1+summary.dat files")
                return
        stacked_data = np.stack(all_data, axis=0)
        summary_avg = stacked_data.mean(axis=0)
        output_file = os.path.join(self.output_path, "Z1+summary_avg.txt")
        np.savetxt(output_file, summary_avg, fmt="%.3f")
        print(f"Finish computing average Z1+summary file in {self.data_path}")


# ---------------------------- use and test ------------------
if __name__ == "__main__":
    base_path = f"/anvil/scratch/x-qwang24/Fall25/OPLS/Tensile"
    systems = ["200_100", "400_50", "800_30"]
    models = ["amorphous", "gel"]
    temperatures = [100, 200, 300]
    rates = ["1e8", "1e9", "1e10"]
    data_paths = []

    for system in systems:
        for model in models:
            for temp in temperatures:
                for rate in rates:
                    data_path = f"{base_path}/{system}/{model}/{temp}/{rate}"
                    data_paths.append(data_path)

    for data_path in data_paths:
        data_path = data_path
        n_para = 10
        output_path = data_path
        avg_z1_sum = Z1SumAvg(data_path, n_para, output_path)
        avg_z1_sum.compute_z1_sum_avg()      