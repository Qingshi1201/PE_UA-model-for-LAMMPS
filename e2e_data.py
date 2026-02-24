# --------------------------------- imports and settings ----------------------
import os
import numpy as np


# --------------------------------- class definition and methods ----------------------
"""
method 1***:
    extract linear components 
    {molecule_1: {monomer_1: [x, y, z],
                  monomer_2: [x, y, z],
                  ...},
     molecule_2: {monomer_1: [x, y, z],
                  monomer_2: [x, y, z],
                  ...},
          .
          .        
          .
    }

method 2:
    print the minimun polymer chain ID

method 3:
    feature 1
    compute end-to-end distances list
"""

class E2E:
    """end-to-end distance class that collects linear chains and compute e2e distance distributon and Kuhn length"""
    def __init__(self, path, filename, exclude_types):
        self.path = path
        self.filename = filename
        self.exclude_types = exclude_types


    def load_data(self):
        """load data file from a specified path"""
        in_atoms = False
        coordinates = {}
        data_file = os.path.join(self.path, self.filename)
        with open(data_file) as model:
            for line in model:
                stripped = line.strip()
                # find Atoms section
                if stripped.startswith("Atoms"):
                    in_atoms = True
                    _ = next(model)
                    continue
                if in_atoms:
                    if stripped == "" or stripped.split()[0].isalpha():
                        break
                    # extract molecule and monomer information
                    parts = stripped.split()
                    atom_id = int(parts[0])
                    mol_id = int(parts[1])
                    atom_type = int(parts[2])
                    if atom_type in self.exclude_types: # only extract long chains and ignore crosslinkers
                        continue
                    x, y, z = map(float, parts[4:7])
                    # store molecule ID
                    if mol_id not in coordinates:
                        coordinates[mol_id] = {}
                    coordinates[mol_id][atom_id] = np.array([x, y, z])   
        for mol_id, monomers in coordinates.items():
            coordinates[mol_id] = dict(sorted(monomers.items()))
        return coordinates
        

    def show_min_molid(self):
        """print minimum polymer chain ID"""
        coordinates = self.load_data()
        for mol_id in coordinates.keys():
            if mol_id == min(coordinates.keys()):
                print(f"The polymer chain index starts from {mol_id}")


    def compute_ree(self):
        """compute end-to-end distances and characteristic ratio"""
        coordinates = self.load_data()
        squared_rees = []
        rees = []
        for mol_id, monomers in coordinates.items():
            monomer_list = [monomers[i] for i in sorted(monomers.keys())]
            first_pos = monomer_list[0]
            last_pos = monomer_list[-1]
            ree_vec = last_pos - first_pos
            ree = np.linalg.norm(ree_vec)
            squared_ree = ree**2
            rees.append(ree)
            squared_rees.append(squared_ree)
        avg_sq_ree = np.mean(squared_rees)
        avg_ree = np.mean(rees)
        return avg_ree, avg_sq_ree, rees, squared_rees




# --------------------------------- test methods and class -------------------
path = "/Volumes/HardDevice/Fall25/Data/OPLS/Tg/test2/post/data/geldata/p3"
filename = "g_400.data"

if __name__ == "__main__":
    model = E2E(path=path, filename=filename, exclude_types=[2])
    model.load_data()
    coordinates = model.load_data()
    min_id = model.show_min_molid()
    print(len(coordinates))
    #print(coordinates[1])
    squared_rees = model.compute_ree()







            
        


