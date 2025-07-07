# ---------------------- imports and settings ----------------
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import random
from random import sample
from scipy.spatial.distance import cdist


# ---------------------- chain conformation parameters --------
d = 1.42
mass = 14
density = 1


# ---------------------- crosslinking parameters --------------
crosslink_density = 0.1
min_bond_length = 0
max_bond_length = 7
max_bond_per_bead = 3


# ---------------------- polydisperse chain lengths -----------
mol_length_distribution = {
    30: 10, # chain length - chain number
    40: 20,
    50: 30,
    60: 30,
    70: 10,
}

total_mols = sum(mol_length_distribution.values())
total_beads = sum(length * count for length, count in mol_length_distribution.items())

# store chain lengths and every chain id in the system
mol_lengths = []
mol_start_indices = []

current_index = 0
for length, count in sorted(mol_length_distribution.items()):
    for _ in range(count):
        mol_lengths.append(length)
        mol_start_indices.append(current_index)
        current_index += length


# ---------------------- box size ---------------------
lx = ly = lz = ((total_beads*mass)/density)**(1/3)


# --------------------- generate polymer chains ---------------
# mol-ID of every bead
mol = []
for mol_id in range(total_mols):
    mol_length = mol_lengths[mol_id]
    for bead_id in range(mol_length):
        mol.append(mol_id+1)

# coordinates of beads along every chain
x = np.zeros(total_beads)
y = np.zeros(total_beads)
z = np.zeros(total_beads)

bead_index = 0
for mol_id in range(total_mols):
    mol_length = mol_lengths[mol_id]

    x[bead_index] = random.randint(-int(lx/2), int(lx/2))
    y[bead_index] = random.randint(-int(ly/2), int(ly/2))
    z[bead_index] = random.randint(-int(lz/2), int(lz/2))

    for i in range(1, mol_length):
        val = random.randint(1, 6)
        if val == 1:
            x[bead_index+i] = x[bead_index+i-1] + d 
            y[bead_index+i] = y[bead_index+i-1]
            z[bead_index+i] = z[bead_index+i-1]
        elif val == 2:
            x[bead_index+i] = x[bead_index+i-1] 
            y[bead_index+i] = y[bead_index+i-1] + d
            z[bead_index+i] = z[bead_index+i-1]
        elif val == 3:
            x[bead_index+i] = x[bead_index+i-1]
            y[bead_index+i] = y[bead_index+i-1] 
            z[bead_index+i] = z[bead_index+i-1] + d
        elif val == 4:
            x[bead_index+i] = x[bead_index+i-1] - d
            y[bead_index+i] = y[bead_index+i-1]
            z[bead_index+i] = z[bead_index+i-1]
        elif val == 5:
            x[bead_index+i] = x[bead_index+i-1]
            y[bead_index+i] = y[bead_index+i-1] - d
            z[bead_index+i] = z[bead_index+i-1]
        else:
            x[bead_index+i] = x[bead_index+i-1]
            y[bead_index+i] = y[bead_index+i-1] 
            z[bead_index+i] = z[bead_index+i-1] - d
    
    bead_index += mol_length

# Apply boundary conditions
for i in range(len(x)):
    if x[i] > lx/2:
        x[i] = lx-x[i]
    elif x[i] < -lx/2:
        x[i] = -lx-x[i]
    else:
        x[i] = x[i]

for i in range(len(y)):
    if y[i] > ly/2:
        y[i] = ly-y[i]
    elif y[i] < -ly/2:
        y[i] = -ly-y[i]
    else:
        y[i] = y[i]

for i in range(len(z)):
    if z[i] > lz/2:
        z[i] = lz-z[i]
    elif z[i] < -lz/2:
        z[i] = -lz-z[i]
    else:
        z[i] = z[i]


# --------------------- randomly select half of chains to crosslink --------------
#mol_indices = list(range(total_mols))
#half_mols = total_mols // 2
selected_mols = sample(list(range(total_mols)), total_mols//2)

bead_types = np.ones(total_beads, dtype=int)
type2_beads = []
type3_beads = []

bead_index = 0
for mol_id in range(total_mols):
    mol_length = mol_lengths[mol_id]
    first_bead_id = bead_index
    last_bead_id = bead_index + mol_length - 1
    if mol_id in selected_mols:
        bead_types[first_bead_id] = 2
        bead_types[last_bead_id] = 2
        type2_beads.extend([first_bead_id, last_bead_id])
    else:
        bead_types[first_bead_id] = 3
        bead_types[last_bead_id] = 3
        type3_beads.extend([first_bead_id, last_bead_id])
    bead_index += mol_length


# --------------------- crosslink by density ---------------
def crosslink_by_density(
        type2_beads, type3_beads, x, y, z,
        target_density, min_length, max_length, max_bonds
):
    # basic output
    print("=== CROSSLINK ANALYSIS ===")   
    print(f"Target crosslink density: {target_density}")
    print(f"Bonding distance range: {min_length} - {max_length} angstroms") 

    # calculate crosslinks
    total_crosslinkable = len(type2_beads) + len(type3_beads)
    target_crosslinked_beads = int(total_crosslinkable * target_density)

    # calculate and check all distances between crosslinkable beads
    type2_coords = np.column_stack((x[type2_beads], y[type2_beads], z[type2_beads]))
    type3_coords = np.column_stack((x[type3_beads], y[type3_beads], z[type3_beads]))
    distances = cdist(type2_coords, type3_coords)

    # find all possible bonds
    possible_bonds = []
    for i in range(len(type2_beads)):
        for j in range(len(type3_beads)):
            dist = distances[i, j]
            if min_length <= dist <= max_length:
                possible_bonds.append((dist, type2_beads[i], type3_beads[j]))
    print("\n=== Total number of possible bonds ===")
    print(f"Found {len(possible_bonds)} possible crosslinking bonds within length constraint")
    if len(possible_bonds) == 0:
        print("WARNING: No possible found in the box to crosslink")
        return []
    
    # add crosslinks
    possible_bonds.sort(key=lambda x: x[0])
    crosslinks = []
    bond_count = {}
    crosslinked_beads = set()
    for dist, bead1, bead2 in possible_bonds:
        if len(crosslinked_beads) >= target_crosslinked_beads:
            break
        if bond_count.get(bead1, 0) >= max_bonds or bond_count.get(bead2, 0) >= max_bonds:
            continue
        if (bead1, bead2) in crosslinks or (bead2, bead1) in crosslinks:
            continue
        crosslinks.append((bead1, bead2))
        bond_count[bead1] = bond_count.get(bead1, 0) + 1
        bond_count[bead2] = bond_count.get(bead2, 0) + 1
        crosslinked_beads.add(bead1)
        crosslinked_beads.add(bead2)

    actual_crosslink_density = len(crosslinked_beads) / total_crosslinkable
    print("\n=== ACTUAL CROSSLINK INFORMATION ===")
    print(f"Actual crosslink density: {actual_crosslink_density:.3f}")
    print(f"Crosslinked beads: {len(crosslinked_beads)} out of {total_crosslinkable}")

    # show bond distribution 
    bond_distribution = {}
    for count in bond_count.values():
        bond_distribution[count] = bond_distribution.get(count, 0) + 1

    print(f"Bond distribution per bead")
    for bonds, beads in sorted(bond_distribution.items()):
        print(f"{bonds} bonds: {beads} beads")

    # end of function
    return crosslinks


# --------------------- generate crosslinks ------------------
crosslinks = crosslink_by_density(type2_beads, type3_beads, x, y, z, crosslink_density, 
                                  min_bond_length, max_bond_length, max_bond_per_bead)
        

# --------------------- analyze topological information ------
# find chain and position of beads on the chain
def get_mol_info(bead_id):
    running_total = 0
    for mol_id, mol_length in enumerate(mol_lengths):
        if bead_id < running_total + mol_length:
            position = bead_id - running_total
            return mol_id, position, mol_length
        running_total += mol_length
    return None

# count bonds
chain_bonds = sum(length-1 for length in mol_lengths)
crosslink_bonds = len(crosslinks)
total_bonds = chain_bonds + crosslink_bonds

# count angles
chain_angles = sum(max(0, length-2) for length in mol_lengths)
crosslink_angles = len(crosslinks) * 2
total_angles = chain_angles + crosslink_angles

# count dihedrals
chain_dihedrals = sum(max(0, length-3) for length in mol_lengths)
crosslink_dihedrals = len(crosslinks)
total_dihedrals = chain_dihedrals + crosslink_dihedrals


# --------------------- write data file --------------
dist_str = "_".join(f"{length}x{count}" for length, count in sorted(mol_length_distribution.items()))
filename = f"polydisperse_{dist_str}_crosslink_{crosslink_density:.2f}.data"

with open(filename, "w") as LAMMPS:
    # header line
    LAMMPS.write("Random bead-spring crosslink polymer model\n\n")

    # counts
    LAMMPS.write("{} atoms\n".format(total_beads))
    LAMMPS.write("{} bonds\n".format(total_bonds))
    LAMMPS.write("{} angles\n".format(total_angles))
    LAMMPS.write("{} dihedrals\n\n".format(total_dihedrals))

    # types
    LAMMPS.write("{} atom types\n".format(3))
    LAMMPS.write("{} bond types\n".format(1))
    LAMMPS.write("{} angle types\n".format(1))
    LAMMPS.write("{} dihedral types\n\n".format(1))

    # box size
    LAMMPS.write("{} {} xlo xhi\n".format(-lx/2, lx/2))
    LAMMPS.write("{} {} ylo yhi\n".format(-ly/2, ly/2))
    LAMMPS.write("{} {} zlo zhi\n\n".format(-lz/2, lz/2))

    # masses
    LAMMPS.write("Masses\n\n")
    LAMMPS.write("{} {} # CH2\n".format(1, mass))
    LAMMPS.write("{} {} # Type2_crosslinker\n".format(2, mass))
    LAMMPS.write("{} {} # Type3_crosslinker\n\n".format(3, mass))

    # atom section 
    LAMMPS.write("Atoms #full\n\n")
    for i in range(total_beads):
        LAMMPS.write("{} {} {} {} {} {} {}\n".format(i+1, mol[i], bead_types[i], 0, x[i], y[i], z[i]))

    # bond section
    LAMMPS.write("\nBonds\n\n")
    bond_id = 1
    bead_id = 0
    # chain bonds
    for mol_id in range(total_mols):
        mol_length = mol_lengths[mol_id]
        for j in range(mol_length-1):
            LAMMPS.write("{} {} {} {}\n".format(bond_id, 1, bead_id+j+1, bead_id+j+2))
            bond_id += 1
        bead_id += mol_length
    # crosslink bonds
    for bead1, bead2 in crosslinks:
        LAMMPS.write("{} {} {} {}\n".format(bond_id, 1, bead1+1, bead2+1))
        bond_id += 1

    # angle section
    LAMMPS.write("\nAngles\n\n")
    angle_id = 1
    bead_id = 0
    # chain angles
    for mol_id in range(total_mols):
        mol_length = mol_lengths[mol_id]
        for j in range(mol_length-2):
            LAMMPS.write("{} {} {} {} {}\n".format(angle_id, 1, j+bead_id+1, j+bead_id+2, j+bead_id+3))
            angle_id += 1
        bead_id += mol_length
    # crosslink angles
    for bead1, bead2 in crosslinks:
        mol1_id, pos1, len1 = get_mol_info(bead1)
        mol2_id, pos2, len2 = get_mol_info(bead2)
        if pos1 == 0:
            neighbor1 = bead1 + 1
        else:
            neighbor1 = bead1 - 1
        if pos2 == 0:
            neighbor2 = bead2 + 1
        else:
            neighbor2 = bead2 - 1
        LAMMPS.write("{} {} {} {} {}\n".format(angle_id, 1, neighbor1+1, bead1+1, bead2+1))
        angle_id += 1
        LAMMPS.write("{} {} {} {} {}\n".format(angle_id, 1, bead1+1, bead2+1, neighbor2+1))
        angle_id += 1

    # dihedral section 
    LAMMPS.write("\nDihedrals\n\n")
    dihedral_id = 1
    bead_id = 0
    # chain dihedrals
    for mol_id in range(total_mols):
        mol_length = mol_lengths[mol_id]
        for j in range(mol_length-3):
            LAMMPS.write("{} {} {} {} {} {}\n".format(dihedral_id, 1, j+bead_id+1, j+bead_id+2, j+bead_id+3, j+bead_id+4))
            dihedral_id += 1
        bead_id += mol_length
    # crosslink dihedrals
    for bead1, bead2 in crosslinks:
        mol1_id, pos1, len1 = get_mol_info(bead1)
        mol2_id, pos2, len2 = get_mol_info(bead2)
        if pos1 == 0 and len1 >= 3:
            neighbor1 = bead1 + 1
            second_neighbor1 = bead1 + 2
        elif pos1 == len1-1 and len1 >= 3:
            neighbor1 = bead1 - 1
            second_neighbor1 = bead2 - 2
        else:
            continue
        if pos2 == 0:
            neighbor2 = bead2 + 1
        else:
            neighbor2 = bead2 - 1
        LAMMPS.write("{} {} {} {} {} {}\n".format(dihedral_id, 1, second_neighbor1+1, neighbor1+1, bead1+1, bead2+1))
        dihedral_id += 1










