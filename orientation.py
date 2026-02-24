# --------------------------------- imports and settings ----------------------
import os
import numpy as np


# --------------------------------- class definition and methods ----------------------
"""
TEST FILES
tensile_x.data / tensile.in / 1e9.lammpstrj (from Fall25/OPLS/Tensile/formal5/amorphous/100K/p1)


METHODS
method 1:
    load trajectory file 
method 2:
    load data file
method 3:
    <any specified frame>
    <start frame> - <end frame> / step: # of frames
    compute <cos theta>&<P2>&<P4>&<P6> of polymer chain bonds over frames
method 4: (optional)
    output compute data
    frame strain <cos_theta> <P2> <P4> <P6>

    
BASIC DATA STRUCTURES
lammpstrj file:
<frame 1>
    ITEM: TIMESTEP
    0
    ITEM: NUMBER OF ATOMS
    XXXX
    ITEM: BOX BOUNDS pp pp pp
    xlo xhi
    ylo yhi
    zlo zhi
    ITEM: ATOMS id type xu yu zu vx vy vz fx fy fz
    ...
<frame 2>


DATA DERIVATION
from data file
    dynamic array atom_ID [number of chains, chain length] (testable)
from trajectory file
    dynamic array atom_position [number of chains, chain length] (testable)
from atom_position array
    list slicing --> dyanmic array chain-bond vecotor
from chain-bond vector array
    compute cosine of bond vector with stretch direction unit vector --> ensemble average <cos(angle)>&<P2>&<P4>&<P6>
"""

class Orientation:
    """compute ensemble average bond angle and different orders of orientation parameters"""
    def __init__(self, path, data_file, lammpstrj_file, crosslinker_type, output_file=None):
        self.path = path
        self.data_file = data_file
        self.lammpstrj_file = lammpstrj_file
        self.crosslinker_type = crosslinker_type
        self.output_file = output_file

    
    def load_data(self):
        """read data file and extract ordered chain monomer atom_ID"""
        chains = {}
        in_atoms = False
        data = os.path.join(self.path, self.data_file)
        # extract monomer and chain mapping
        with open(data) as model:
            for line in model:
                stripped = line.strip()
                if stripped.startswith("Atoms"):
                    in_atoms = True
                    _ = next(model)
                    continue
                if in_atoms:
                    if stripped == "" or stripped.split()[0].isalpha():
                        break
                    parts = stripped.split()
                    atom_id = int(parts[0])
                    mol_id = int(parts[1])
                    atom_type = int(parts[2])
                    if atom_type in self.crosslinker_type:
                        continue
                    if mol_id not in chains:
                        chains[mol_id] = []
                    chains[mol_id].append(atom_id)
        chains = {mol_id: sorted(monomer_id) for mol_id, monomer_id in sorted(chains.items())}

        # monomers list into 2D dynamic array
        n_chains = len(chains)
        l_chain = len(chains[1])
        monomer_ID = []
        for monomer_list in chains.values():
            monomer_ID.append(monomer_list)
        monomer_ID = np.array(monomer_ID)
        return monomer_ID


    def collect_frames(self, start_frame=0, end_frame=None, stride=1):
        """collect required frames from trajectory file"""
        # sample single frame information from lammps trajectory
        trajectory = os.path.join(self.path, self.lammpstrj_file)
        n_frames = 0
        frame_id = -1
        frame_lines = []
        with open(trajectory, "r") as trj:
            for line in trj:
                stripped = line.strip()
                if stripped.startswith("ITEM: TIMESTEP"):
                    n_frames += 1
                    frame_id += 1

                    if frame_lines:
                        if frame_id-1 >= start_frame and (frame_id-1-start_frame)%stride == 0:
                            yield frame_lines
                        frame_lines = []
                    if end_frame is not None and frame_id > end_frame:
                        break
                frame_lines.append(line)
            if frame_lines:
                if frame_id >= start_frame and (frame_id-start_frame) % stride == 0:
                    yield frame_lines
        frame_list = list(frame_lines)
        return frame_list


    def extract_bond_vectors(self, start_frame=0, end_frame=None, stride=1):
        """extract coordinates of corresponding monoomers"""
        monomer_ID = self.load_data()
        frame_lines = self.collect_frames(start_frame=start_frame, end_frame=end_frame, stride=stride)
        frame_lists = list(frame_lines)
        n_chains = len(monomer_ID)
        l_chain = len(monomer_ID[0])

        # assign correct coordinates of monomers into 2D array
        monomer_coordinates = np.zeros((n_chains, l_chain, 3), dtype=float)
        for frame_list in frame_lists:
            atom_section = frame_list[9:]

            monomer_coords_dict = {}
            for line in atom_section:
                parts = line.split()
                monomer_id = int(parts[0])
                monomer_coord = list(map(float, parts[2:5]))
                monomer_coords_dict[monomer_id] = monomer_coord

            for i, row in enumerate(monomer_ID):
                for j, monomer_id in enumerate(row):
                    if monomer_id in monomer_coords_dict:
                        monomer_coordinates[i, j, :] = monomer_coords_dict[monomer_id]
            bond_vecotors = monomer_coordinates[:, 1:, :] - monomer_coordinates[:, :-1, :]
            yield bond_vecotors
        return bond_vecotors
    

    def compute_parameters(self, start_frame=0, end_frame=None, stride=1, output=None):
        """bond vectors, compute orientation parameters and output data"""
        bond_vectors = self.extract_bond_vectors(start_frame, end_frame, stride)
        vectors_list = list(bond_vectors)

        # compute cosine angle of stretch and bonds and orientation parameters
        principal_v = np.array([1,0,0])
        cos_dict = {}
        cos_power2_dict = {}
        cos_power4_dict = {}
        cos_power6_dict = {}
        for frame_id, frame in enumerate(vectors_list):
            cos_dict[frame_id] = []
            cos_power2_dict[frame_id] = []
            cos_power4_dict[frame_id] = []
            cos_power6_dict[frame_id] = []
            for chain in frame:
                for bond in chain:
                    dot_product = np.dot(bond, principal_v)
                    norm_bond = np.linalg.norm(bond)
                    norm_principal = np.linalg.norm(principal_v)
                    cos_theta = dot_product / (norm_bond*norm_principal)
                    cos_dict[frame_id].append(cos_theta)
                    cos_power2_dict[frame_id].append(cos_theta**2)
                    cos_power4_dict[frame_id].append(cos_theta**4)
                    cos_power6_dict[frame_id].append(cos_theta**6)

        # make ensemble average and compute P parameters
        avg_cos_dict = {} # averaged cos2, cos4, cos6 of every single frame {frame_ID: [avg_cos2, avg_cos4, avg_cos6],}
        frame_p_dict = {} # {frame_id: [P2, P4, P6]}
        for frame_id in cos_power2_dict.keys():
            avg_cos_dict[frame_id] = []
        for frame_id, cos_value_list in avg_cos_dict.items():
            cos_value_list.append(np.array(cos_power2_dict[frame_id]).mean())
            cos_value_list.append(np.array(cos_power4_dict[frame_id]).mean())
            cos_value_list.append(np.array(cos_power6_dict[frame_id]).mean())
        
        for frame_id in avg_cos_dict.keys():
            frame_p_dict[frame_id] = []
        for frame_id, p_values_list in frame_p_dict.items():
            P2 = (3*avg_cos_dict[frame_id][0] - 1) / 2
            P4 = (35*avg_cos_dict[frame_id][1] - 30*avg_cos_dict[frame_id][0] + 3) / 8
            P6 = (231*avg_cos_dict[frame_id][2] - 315*avg_cos_dict[frame_id][1] + 105*avg_cos_dict[frame_id][0] - 5) / 16
            p_values_list.append(P2)
            p_values_list.append(P4)
            p_values_list.append(P6)

        # optionally output data in txt file
        if output != None:
            output_file = os.path.join(self.path, output)
            with open(output_file, "w") as P:
                P.write(f"frame_id <P2> <P4> <P6>\n")
                for frame_id, p_values_list in frame_p_dict.items():
                    P.write(f"{frame_id} {p_values_list[0]} {p_values_list[1]} {p_values_list[2]}\n")
        
        return frame_p_dict
    

# --------------------------------- test methods and features ----------------------
if __name__ == "__main__":
    path = "/Users/qingshiwang/Research/Python/Orientation"
    data_file = "tensile_x.data"
    lammpstrj_file = "1e9.lammpstrj"
    crosslinker_type = [2]
    start_frame = 0
    stride = 1

    model = Orientation(
        path=path, data_file=data_file, lammpstrj_file=lammpstrj_file, 
        crosslinker_type=crosslinker_type
    )
    chains = model.load_data()
    bond_vectors = model.extract_bond_vectors(start_frame=start_frame, stride=stride)
    frame_lines = model.compute_parameters(start_frame=start_frame, stride=stride, output="p_stretch.txt")
    print("All done")
   
