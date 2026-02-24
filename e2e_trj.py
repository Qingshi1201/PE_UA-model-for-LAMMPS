# --------------------------------- imports and settings ----------------------
import os
import numpy as np


# --------------------------------- class definition and methods ----------------------
"""
REQUIRED DATA/FILE
Z1+summary.dat (gel/250K/1e10)

METHODS
method 1: (from multiple parallel simulations)
    load data file &&
    extract required data columns
method 2:
    compute averaged distances by frame with standard deviation &&
    output avg file ree_a_${temperature}_${rate}.txt
"""

class E2EFrame:
    """compute frame-level average end-to-end distances"""
    def __init__(self, base_path, n_paras, z1_file):
        self.base_path = base_path # base path contains parallel p folders
        self.n_paras = n_paras
        self.z1_file = z1_file

    def compute_frame_e2e(self):
        """extract data, compute and write output file"""
        z1_paths = []
        for i in range(1, self.n_paras+1):
            para_path = f"p{i}"
            z1_path = os.path.join(self.base_path, para_path, self.z1_file)
            z1_paths.append(z1_path)

        ms_rees = []
        for z1_path in z1_paths:
            z1_data = np.loadtxt(z1_path)
            frames = z1_data[:, 0]
            ms_ree = z1_data[:, 3]
            ms_rees.append(ms_ree)
        ms_rees = np.array(ms_rees)
        ms_ree_avg = np.mean(ms_rees, axis=0)
        ms_ree_std = np.std(ms_rees, axis=0)

        output_file = "frame_rees_avg.txt"
        output = os.path.join(self.base_path, output_file)
        header = "frame_ID Ree_avg Ree_std"
        data = np.column_stack([frames, ms_ree_avg, ms_ree_std])
        fmt = ["%d"] + ["%.3f"] * 2
        np.savetxt(output, data, fmt=fmt, header=header)

        return None    


# --------------------------------- usage and test ----------------------
if __name__ == "__main__":
    parent_path = f"/anvil/scratch/x-qwang24/Fall25/OPLS/Tensile/"
    chains = ["200_100", "400_50", "800_30"]
    model_types = ["amorphous", "gel"]
    temperatures = ["200K", "300K"]
    rates = ["1e8", "1e9", "1e10"]
    n_paras = 10
    z1_file = "Z1+summary.dat"
    base_paths = []

    for chain in chains:
        for model in model_types:
            for temperature in temperatures:
                for rate in rates:
                    base_path = f"{parent_path}/{chain}/{model}/{temperature}/{rate}"
                    base_paths.append(base_path)

    for base_path in base_paths:
        n_paras = 10
        z1_file = "Z1+summary.dat"
        frame_ree = E2EFrame(base_path=base_path, n_paras=n_paras, z1_file=z1_file)
        frame_ree.compute_frame_e2e()







