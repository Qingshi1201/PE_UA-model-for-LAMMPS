# --------------------- imports and settings -------------
import os 
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import random
from random import sample
import numpy as np 
from scipy.spatial.distance import cdist
import math


# ------------------- single chain parameters ------------
c_mass = 12
h_mass = 1
cc_bond = 1.42
ch_bond = 1.07
density = 1


# ------------------ crosslink parameters -----------------
crosslink_density = 0.7 # range from 0 to 1
min_bond_distance = 0 # accepted distance to bond carbon atoms
max_bond_distance = 10
max_crosslinks_per_carbon = 3 # max crosslink bonds per carbon atom

length_distribution = {
    100: 20, # chain_length: chain_number
    110: 10,
    120: 10,
    130: 10,
}

total_chains = sum(length_distribution.values())
total_carbons = sum(chain*count for chain, count in length_distribution.items())


chain_lengths = []
chain_start_indices = []

current_index = 0
for length, count in sorted(length_distribution.items()):
    for _ in range(count):
        chain_lengths.append(length)
        chain_start_indices.append(current_index)
        current_index += length


# ------------------- box size ----------------------------
lx = ly = lz = ((c_mass*total_carbons)/density)**(1/3)


# ------------------- generate coordinates of all carbon atoms ----------
# generate mol-ID
mol = []
for chain_id in range(total_chains):
    chain_length = chain_lengths[chain_id]
    for carbon_id in range(chain_length):
        mol.append(chain_id+1)

# generate coordinates of carbons
x = np.zeros(total_carbons)
y = np.zeros(total_carbons)
z = np.zeros(total_carbons)

# loop over chains
carbon_index = 0
for chain_id in range(total_chains):
    chain_length = chain_lengths[chain_id]
    x[carbon_index] = random.randint(-int(lx/2), int(lx/2))
    y[carbon_index] = random.randint(-int(ly/2), int(ly/2))
    z[carbon_index] = random.randint(-int(lz/2), int(lz/2))

    # loop over carbons along chains
    for i in range(1, chain_length):
        val = random.randint(1, 6)
        if val == 1:
            x[i+carbon_index] = x[i+carbon_index-1] + cc_bond
            y[i+carbon_index] = y[i+carbon_index-1]
            z[i+carbon_index] = z[i+carbon_index-1]
        elif val == 2:
            x[i+carbon_index] = x[i+carbon_index-1]
            y[i+carbon_index] = y[i+carbon_index-1] + cc_bond
            z[i+carbon_index] = z[i+carbon_index-1]
        elif val == 3:
            x[i+carbon_index] = x[i+carbon_index-1]
            y[i+carbon_index] = y[i+carbon_index-1]
            z[i+carbon_index] = z[i+carbon_index-1] + cc_bond
        elif val == 4:
            x[i+carbon_index] = x[i+carbon_index-1] - cc_bond
            y[i+carbon_index] = y[i+carbon_index-1]
            z[i+carbon_index] = z[i+carbon_index-1]
        elif val == 5:
            x[i+carbon_index] = x[i+carbon_index-1]
            y[i+carbon_index] = y[i+carbon_index-1] - cc_bond
            z[i+carbon_index] = z[i+carbon_index-1]
        else:
            x[i+carbon_index] = x[i+carbon_index-1]
            y[i+carbon_index] = y[i+carbon_index-1]
            z[i+carbon_index] = z[i+carbon_index-1] - cc_bond
    carbon_index += chain_length


# ---------------- boundary condition ----------------------
# x direction
for i in range(len(x)):
    if x[i] > lx/2:
        x[i] = lx - x[i]
    elif x[i] < -lx/2:
        x[i] = -lx - x[i]
    else:
        x[i] = x[i]

# y direction
for i in range(len(y)):
    if y[i] > ly/2:
        y[i] = ly - y[i]
    elif y[i] < -ly/2:
        y[i] = -ly - y[i]
    else:
        y[i] = y[i]

# z direction 
for i in range(len(z)):
    if z[i] > lz/2:
        z[i] = lz - z[i]
    elif z[i] < -lz/2:
        z[i] = -lz - z[i]
    else:
        z[i] = z[i]


# ---------------- randomly select chains ----------------
half_chains = total_chains // 2
selected_chains = sample(list(range(total_chains)), half_chains)

carbon_types = np.ones(total_carbons, dtype=int)
type2_carbons = []
type3_carbons = []

carbon_index = 0
for chain_id in range(total_chains):
    chain_length = chain_lengths[chain_id] # calculate id of first carbon of chains
    first_carbon = carbon_index # calculate id of last carbon of chains
    last_carbon = carbon_index + chain_length - 1

    if chain_id in selected_chains:
        carbon_types[first_carbon] = 2
        carbon_types[last_carbon] = 2
        type2_carbons.extend([first_carbon, last_carbon])
    else:
        carbon_types[first_carbon] = 3
        carbon_types[last_carbon] = 3
        type3_carbons.extend([first_carbon, last_carbon]) 
    carbon_index += chain_length
    


# ---------------- crosslink by density ----------------
def crosslink_by_density(type2_carbons, type3_carbons, x, y, z,
                         target_density, min_distance, max_distance, max_crosslinks_per_carbon):
    '''
    target_density = crosslink_density
    min_distance = 0
    max_distance = 7
    max_bonds_per_carbon = 3
    '''
    total_crosslinkable = len(type2_carbons) + len(type3_carbons)
    target_crosslinked = int(total_crosslinkable * target_density)
    #print(f"crosslinkable {total_crosslinkable}")
    #print(f"target crosslink {target_crosslinked}")

    type2_coords = np.column_stack((x[type2_carbons], y[type2_carbons], z[type2_carbons]))
    type3_coords = np.column_stack((x[type3_carbons], y[type3_carbons], z[type3_carbons]))
    distances = cdist(type2_coords, type3_coords)

    possible_crosslinks = []
    for i in range(len(type2_carbons)):
        for j in range(len(type3_carbons)):
            distance = distances[i, j]
            if min_distance <= distance <= max_distance:
                possible_crosslinks.append((distance, type2_carbons[i], type3_carbons[j]))

    if len(possible_crosslinks) == 0:
        print("WARNING: No crosslinks found in the box")
        return []
    
    possible_crosslinks.sort(key=lambda x: x[0])
    final_crosslinks = []
    crosslink_count = {}
    crosslinked_carbons = set()

    for distance, cros_carbon1, cros_carbon2 in possible_crosslinks:
        if len(crosslinked_carbons) >= target_crosslinked:
            break
        if crosslink_count.get(cros_carbon1, 0)>=max_crosslinks_per_carbon or crosslink_count.get(cros_carbon2, 0)>=max_crosslinks_per_carbon:
            continue
        if (cros_carbon1, cros_carbon2) in possible_crosslinks or (cros_carbon2, cros_carbon1) in possible_crosslinks:
            continue

        final_crosslinks.append((cros_carbon1, cros_carbon2))
        crosslink_count[cros_carbon1] = crosslink_count.get(cros_carbon1, 0) + 1
        crosslink_count[cros_carbon2] = crosslink_count.get(cros_carbon2, 0) + 1
        crosslinked_carbons.add(cros_carbon1)
        crosslinked_carbons.add(cros_carbon2)

    actual_crosslink_density = len(crosslinked_carbons) / total_crosslinkable
    print("=== Crosslink Results ===")
    print(f"Target crosslink density: {crosslink_density}")
    print(f"Actual crosslink density {actual_crosslink_density:.3f}")
    print(f"{len(crosslinked_carbons)} out of {total_crosslinkable} carbons have been crosslinked")

    # get information of crosslink bonds on every crosslinked carbons
    cros_bonds_distribution = {}
    for count in crosslink_count.values():
        cros_bonds_distribution[count] = cros_bonds_distribution.get(count, 0) + 1
        
    print(f"Crosslink bonds distribution")
    for bonds, carbons in sorted(cros_bonds_distribution.items()):
        print(f"{bonds} crosslinks: {carbons} carbons")

    # end of function
    return final_crosslinks

# ---------------- use the function to generate crosslinks --------------------
crosslinks = crosslink_by_density(type2_carbons, type3_carbons, x, y, z, 
                                  crosslink_density, min_bond_distance, max_bond_distance, max_crosslinks_per_carbon)


# ---------------- get number of bonds between carbon atoms -------------------
def count_carbon_bonds(): 
    carbon_bonds = [0] * total_carbons
    # count bonds along the chain 
    carbon_id = 0
    for chain in range(total_chains):
        chain_length = chain_lengths[chain]
        for i in range(chain_length-1):
            carbon_bonds[i+carbon_id] += 1
            carbon_bonds[i+carbon_id+1] += 1
        carbon_id += chain_length
    # count crosslink bonds
    for cros_carbon1, cros_carbon2 in crosslinks:
        carbon_bonds[cros_carbon1] += 1
        carbon_bonds[cros_carbon2] += 1
    # end of function
    return carbon_bonds


# ---------------- add hydrogen atoms accordingly -----------------------------
def add_hydrogens(carbon_bonds, x, y, z):
    # lists to save data
    h_molecule = []
    h_coords = []
    h_carbon_bonds = []
    # add hydrogen numbers
    total_hydrogens = 0
    h_distribution = {
        0: 0, # number of hydrogens on a carbon: number of carbons 
        1: 0,
        2: 0,
        3: 0
    }
    for carbon in range(total_carbons):
        cc_bonds = carbon_bonds[carbon]
        h_count = max(0, 4-cc_bonds)
        h_distribution[h_count] += 1
    # generate hydrogen coordinates
        if h_count > 0:
            carbon_coord = np.array([x[carbon], y[carbon], z[carbon]])

            for h_id in range(h_count):
                theta = random.uniform(0, 2*math.pi)
                phi = random.uniform(0, math.pi)
                h_x = carbon_coord[0] + ch_bond * math.sin(phi) * math.cos(theta)
                h_y = carbon_coord[1] + ch_bond * math.sin(phi) * math.sin(theta)
                h_z = carbon_coord[2] + ch_bond * math.cos(phi)

                h_coords.append([h_x, h_y, h_z])
                h_molecule.append(mol[carbon])
                h_carbon_bonds.append(carbon)
                total_hydrogens += 1
    # output information
    print("\n=== Hydrogen Addition ===")
    print(f"Added {total_hydrogens} hydrogens")
    print("Hydrogen distribution")
    for h_count, carbon_count in h_distribution.items():
        print(f"{h_count} hydrogen(s) {carbon_count} carbons")
    # end of function
    return np.array(h_coords), h_molecule, h_carbon_bonds, total_hydrogens


# ---------------- function for chain information given a carbon in crosslinks ---
def get_chain_info(carbon_id):
    # calculate number of C-C bonds on every carbon
    running_total = 0
    for chain_id, chain_length in enumerate(chain_lengths):
        if carbon_id < chain_length + running_total:
            pos = carbon_id - running_total
            return chain_id, pos, chain_length
        running_total += chain_length
    # end of function
    return None


# ---------------- calculate carbon bonds and add hydrogens accordingly ------
carbon_bonds = count_carbon_bonds()
h_coords, h_molecule, h_carbon_bonds, total_hydrogens = add_hydrogens(carbon_bonds, x, y, z)


# ---------------- information of bonds, angles, dihedrals --------------------
# atom information
total_atoms = total_carbons + total_hydrogens

# bond information
chain_bonds = sum(chain_length-1 for chain_length in chain_lengths)
crosslink_bonds = len(crosslinks)
ch_bonds = total_hydrogens
total_bonds = chain_bonds + crosslink_bonds + ch_bonds

# angle information
# C-C-C angles
chain_angles = sum(chain_length-2 for chain_length in chain_lengths)
crosslink_angles = 2 * len(crosslinks)
# H-C-H & H-C-C angles
hch_angles = 0
hcc_angles = 0
for carbon in range(total_carbons):
    cc_bond = carbon_bonds[carbon]
    h_count = max(0, 4-cc_bond)
    if h_count >= 2:
        hch_angles += (h_count * (h_count-1)) // 2
    if h_count >= 1 and cc_bond >= 1:
        hcc_angles += h_count * cc_bond

total_angles = chain_angles + crosslink_angles + hch_angles + hcc_angles

# dihedral information
chain_dihedrals = sum(chain_length-3 for chain_length in chain_lengths)
crosslink_dihedrals = len(crosslinks)

# Count H-C-C-H and H-C-C-C dihedrals
hcch_dihedrals = 0
hccc_dihedrals = 0

for carbon in range(total_carbons):
    cc_bond = carbon_bonds[carbon]
    h_count = max(0, 4-cc_bond)
    
    if h_count >= 1:
        # Get carbon neighbors
        carbon_neighbors = []
        chain_id, pos, chain_len = get_chain_info(carbon)
        
        # Add chain neighbors
        if pos > 0:
            carbon_neighbors.append(carbon-1)
        if pos < chain_len - 1:
            carbon_neighbors.append(carbon+1)
        
        # Add crosslink neighbors
        for cros_carbon1, cros_carbon2 in crosslinks:
            if cros_carbon1 == carbon:
                carbon_neighbors.append(cros_carbon2)
            elif cros_carbon2 == carbon:
                carbon_neighbors.append(cros_carbon1)
        
        # For each carbon neighbor, count H-C-C-H and H-C-C-C dihedrals
        for neighbor in carbon_neighbors:
            neighbor_cc_bond = carbon_bonds[neighbor]
            neighbor_h_count = max(0, 4-neighbor_cc_bond)
            
            # H-C-C-H dihedrals (H on this carbon, H on neighbor carbon)
            hcch_dihedrals += h_count * neighbor_h_count
            
            # H-C-C-C dihedrals (H on this carbon, C bonded to neighbor carbon)
            neighbor_chain_id, neighbor_pos, neighbor_chain_len = get_chain_info(neighbor)
            neighbor_carbon_neighbors = []
            
            # Add chain neighbors of the neighbor
            if neighbor_pos > 0 and neighbor-1 != carbon:
                neighbor_carbon_neighbors.append(neighbor-1)
            if neighbor_pos < neighbor_chain_len - 1 and neighbor+1 != carbon:
                neighbor_carbon_neighbors.append(neighbor+1)
            
            # Add crosslink neighbors of the neighbor
            for cros_carbon1, cros_carbon2 in crosslinks:
                if cros_carbon1 == neighbor and cros_carbon2 != carbon:
                    neighbor_carbon_neighbors.append(cros_carbon2)
                elif cros_carbon2 == neighbor and cros_carbon1 != carbon:
                    neighbor_carbon_neighbors.append(cros_carbon1)
            
            hccc_dihedrals += h_count * len(neighbor_carbon_neighbors)

total_dihedrals = chain_dihedrals + crosslink_dihedrals + hcch_dihedrals + hccc_dihedrals

print("\n=== Data File Basic Information ===")
print(f"{total_atoms} Atoms")
print(f"{total_bonds} Bonds")
print(f"{total_angles} Angles")
print(f"{total_dihedrals} Dihedrals")


# ---------------- write data file ----------------
filename_str = "_".join(f"{length}x{count}" for length, count in length_distribution.items())
filename = f"pd_{filename_str}_cd_{crosslink_density}.data"
with open(filename, "w") as LAMMPS:
    # header line
    LAMMPS.write("crosslinked polydispersed Bead-spring model\n")

    # counts 
    LAMMPS.write("{} atoms\n".format(total_atoms))
    LAMMPS.write("{} bonds\n".format(total_bonds))
    LAMMPS.write("{} angles\n".format(total_angles))
    LAMMPS.write("{} dihedrals\n\n".format(total_dihedrals))

    # types
    LAMMPS.write("{} atom types\n".format(2))
    LAMMPS.write("{} bond types\n".format(2))
    LAMMPS.write("{} angle types\n".format(3))
    LAMMPS.write("{} dihedral types\n\n".format(3))  # Changed from 1 to 3 types

    # box dimension
    LAMMPS.write("{} {} xlo xhi\n".format(-lx/2, lx/2))
    LAMMPS.write("{} {} ylo yhi\n".format(-ly/2, ly/2))
    LAMMPS.write("{} {} zlo zhi\n\n".format(-lz/2, lz/2))

    # masses
    LAMMPS.write("Masses\n\n")
    LAMMPS.write("{} {} # C\n".format(1, c_mass))
    #LAMMPS.write("{} {} # type2_crosslinkable\n".format(2, c_mass))
    #LAMMPS.write("{} {} # type3_crosslinkable\n".format(3, c_mass))
    LAMMPS.write("{} {} # H\n\n".format(2, h_mass))

    # atom section 
    LAMMPS.write("Atoms #full\n\n") # atom_style: full atom-ID, mol-ID, atom-type, q, x, y, z
    # carbon atoms
    for i in range(total_carbons):
        LAMMPS.write("{} {} {} {} {} {} {}\n".format(i+1, mol[i], 1, 0, x[i], y[i], z[i]))
    # hydrogen atoms
    for i in range(total_hydrogens):
        h_id = i + total_carbons + 1
        LAMMPS.write("{} {} {} {} {} {} {}\n".format(h_id, h_molecule[i], 2, 0, 
                     h_coords[i][0], h_coords[i][1], h_coords[i][2]))

    # bond section
    LAMMPS.write("\nBonds\n\n")
    # C-C bonds
    bond_id = 1
    carbon_id = 0
    for chain in range(total_chains):
        chain_length = chain_lengths[chain]
        for i in range(chain_length-1):
            LAMMPS.write("{} {} {} {}\n".format(bond_id, 1, i+carbon_id+1, i+carbon_id+2))
            bond_id += 1
        carbon_id += chain_length
    for carbon1, carbon2 in crosslinks:
        LAMMPS.write("{} {} {} {}\n".format(bond_id, 1, carbon1+1, carbon2+1))
        bond_id += 1
    # C-H bonds
    for i in range(total_hydrogens):
        carbon_id = h_carbon_bonds[i]
        h_id = i + total_carbons + 1
        LAMMPS.write("{} {} {} {}\n".format(bond_id, 2, carbon_id+1, h_id))
        bond_id += 1


    # angle section
    LAMMPS.write("\nAngles\n\n")
    # C-C-C bonds
    angle_id = 1
    carbon_id = 0
    for chain in range(total_chains):
        chain_length = chain_lengths[chain]
        for i in range(chain_length-2):
            LAMMPS.write("{} {} {} {} {}\n".format(angle_id, 1, carbon_id+i+1, carbon_id+i+2, carbon_id+i+3))
            angle_id += 1
        carbon_id = chain_length
    for carbon1, carbon2 in crosslinks:
        chain_id1, pos1, chain_length1 = get_chain_info(carbon1)
        chain_id2, pos2, chain_length2 = get_chain_info(carbon2)
        if pos1 == 0:
            neighbor1 = carbon1 + 1
        else:
            neighbor1 = carbon1 - 1
        if pos2 == 0:
            neighbor2 = carbon2 + 1
        else:
            neighbor2 = carbon2 - 1
        LAMMPS.write("{} {} {} {} {}\n".format(angle_id, 1, neighbor1+1, carbon1+1, carbon2+1))
        angle_id += 1
        LAMMPS.write("{} {} {} {} {}\n".format(angle_id, 1, carbon1+1, carbon2+1, neighbor2+1))
        angle_id += 1
    # H-C-H & H-C-C bonds
    h_id = 0
    for carbon in range(total_carbons):
        cc_bond = carbon_bonds[carbon]
        h_count = max(0, 4-cc_bond)

        if h_count > 0:
            h_atoms = [total_carbons+h_id+i+1 for i in range(h_count)] # get H-id for every carbon atom
            # H-C-H bonds
            for i in range(len(h_atoms)):
                for j in range(i+1, len(h_atoms)):
                    LAMMPS.write("{} {} {} {} {}\n".format(angle_id, 2, 
                                                           h_atoms[i], carbon+1, h_atoms[j]))
                    angle_id += 1
            h_id += h_count # move to next carbon atom
            # H-C-C bonds
            carbon_neighbors = []
            chain_id, pos, chain_len = get_chain_info(carbon)
            if pos > 0:
                carbon_neighbors.append(carbon-1)
            if pos < chain_len - 1:
                carbon_neighbors.append(carbon+1)
            for cros_carbon1, cros_carbon2 in crosslinks:
                if cros_carbon1 == carbon:
                    carbon_neighbors.append(cros_carbon2)
                elif cros_carbon2 == carbon:
                    carbon_neighbors.append(cros_carbon1)

            for h_atom in h_atoms:
                for carbon_neighbor in carbon_neighbors:
                    LAMMPS.write("{} {} {} {} {}\n".format(angle_id, 3, h_atom, carbon+1, carbon_neighbor+1))
                    angle_id += 1


    # dihedral section 
    LAMMPS.write("\nDihedrals\n\n")
    dihedral_id = 1
    
    # C-C-C-C dihedrals along chains
    carbon_id = 0
    for chain in range(total_chains):
        chain_length = chain_lengths[chain]
        for i in range(chain_length-3):
            LAMMPS.write("{} {} {} {} {} {}\n".format(dihedral_id, 1, i+carbon_id+1, i+carbon_id+2, i+carbon_id+3, i+carbon_id+4))
            dihedral_id += 1
        carbon_id += chain_length
    
    # C-C-C-C dihedrals at crosslinks
    for carbon1, carbon2 in crosslinks:
        chain_id1, pos1, chain_length1 = get_chain_info(carbon1)
        chain_id2, pos2, chain_length2 = get_chain_info(carbon2)
        if pos1 == 0:
            neighbor1 = carbon1 + 1
            second_neigbhor1 = carbon1 + 2
        elif pos1 == chain_length1-1:
            neighbor1 = carbon1 - 1
            second_neigbhor1 = carbon1 - 2
        else:
            continue
        if pos2 == 0:
            neighbor2 = carbon2 + 1
        else:
            neighbor2 = carbon2 - 1
        LAMMPS.write("{} {} {} {} {} {}\n".format(dihedral_id, 1, second_neigbhor1+1, neighbor1+1, carbon1+1, carbon2+1))
        dihedral_id += 1
    
    # H-C-C-H and H-C-C-C dihedrals
    h_id = 0
    for carbon in range(total_carbons):
        cc_bond = carbon_bonds[carbon]
        h_count = max(0, 4-cc_bond)
        
        if h_count > 0:
            h_atoms = [total_carbons+h_id+i+1 for i in range(h_count)]
            h_id += h_count
            
            # Get carbon neighbors
            carbon_neighbors = []
            chain_id, pos, chain_len = get_chain_info(carbon)
            
            # Add chain neighbors
            if pos > 0:
                carbon_neighbors.append(carbon-1)
            if pos < chain_len - 1:
                carbon_neighbors.append(carbon+1)
            
            # Add crosslink neighbors
            for cros_carbon1, cros_carbon2 in crosslinks:
                if cros_carbon1 == carbon:
                    carbon_neighbors.append(cros_carbon2)
                elif cros_carbon2 == carbon:
                    carbon_neighbors.append(cros_carbon1)
            
            # For each carbon neighbor
            for neighbor in carbon_neighbors:
                neighbor_cc_bond = carbon_bonds[neighbor]
                neighbor_h_count = max(0, 4-neighbor_cc_bond)
                
                # Find hydrogen atoms on neighbor carbon
                neighbor_h_atoms = []
                temp_h_id = 0
                for temp_carbon in range(neighbor):
                    temp_cc_bond = carbon_bonds[temp_carbon]
                    temp_h_count = max(0, 4-temp_cc_bond)
                    temp_h_id += temp_h_count
                
                if neighbor_h_count > 0:
                    neighbor_h_atoms = [total_carbons+temp_h_id+i+1 for i in range(neighbor_h_count)]
                
                # H-C-C-H dihedrals (type 2)
                for h_atom in h_atoms:
                    for neighbor_h_atom in neighbor_h_atoms:
                        LAMMPS.write("{} {} {} {} {} {}\n".format(dihedral_id, 2, h_atom, carbon+1, neighbor+1, neighbor_h_atom))
                        dihedral_id += 1
                
                # H-C-C-C dihedrals (type 3)
                # Get carbon neighbors of the neighbor carbon
                neighbor_chain_id, neighbor_pos, neighbor_chain_len = get_chain_info(neighbor)
                neighbor_carbon_neighbors = []
                
                # Add chain neighbors of the neighbor
                if neighbor_pos > 0 and neighbor-1 != carbon:
                    neighbor_carbon_neighbors.append(neighbor-1)
                if neighbor_pos < neighbor_chain_len - 1 and neighbor+1 != carbon:
                    neighbor_carbon_neighbors.append(neighbor+1)
                
                # Add crosslink neighbors of the neighbor
                for cros_carbon1, cros_carbon2 in crosslinks:
                    if cros_carbon1 == neighbor and cros_carbon2 != carbon:
                        neighbor_carbon_neighbors.append(cros_carbon2)
                    elif cros_carbon2 == neighbor and cros_carbon1 != carbon:
                        neighbor_carbon_neighbors.append(cros_carbon1)
                
                for h_atom in h_atoms:
                    for neighbor_neighbor in neighbor_carbon_neighbors:
                        LAMMPS.write("{} {} {} {} {} {}\n".format(dihedral_id, 3, h_atom, carbon+1, neighbor+1, neighbor_neighbor+1))
                        dihedral_id += 1

print("\n=== All done ===")