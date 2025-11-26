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
reaction_extent = 0.6
min_bond_length = 0
max_bond_length = 15


# ---------------------- polydisperse chain lengths -----------
l_chain = 100
n_chain = 100
f_chain = 2
max_t2_bonds = 1
l_angent = 1
n_agent = 100
f_agent = 4
max_t3_bonds = 4

mol_length_distribution = {
    l_chain: n_chain, # polymer chain: number
    l_angent: n_agent, # crosslinking agent: number
}

total_mols = sum(mol_length_distribution.values())
total_beads = sum(length * count for length, count in mol_length_distribution.items())

# store chain lengths and every chain id in the system
mol_lengths = []
mol_start_indices = []

current_index = 0
for length, count in mol_length_distribution.items():
    for _ in range(count):
        mol_lengths.append(length)
        mol_start_indices.append(current_index)
        current_index += length


# ---------------------- box size ---------------------
lx = ly = lz = ((total_beads*mass)/density)**(1/3)


# --------------------- generate polymer chains ---------------
# mol-ID of every bead
mol_ids = []
for mol_id in range(total_mols):
    mol_length = mol_lengths[mol_id]
    for bead_id in range(mol_length):
        mol_ids.append(mol_id+1)

# coordinates of beads along every chain
x = np.zeros(total_beads)
y = np.zeros(total_beads)
z = np.zeros(total_beads)

# possible directions
directions = {
    1: (d, 0, 0),
    2: (0, d, 0),
    3: (0, 0, d),
    4: (-d, 0, 0),
    5: (0, -d, 0),
    6: (0, 0, -d),
}

# reversible pairs
reversible_pairs = {
    1: 4,
    2: 5,
    3: 6,
    4: 1,
    5: 2,
    6: 3,
}

bead_index = 0
for mol_id in range(total_mols):
    mol_length = mol_lengths[mol_id]

    x[bead_index] = random.randint(-int(lx/2), int(lx/2))
    y[bead_index] = random.randint(-int(ly/2), int(ly/2))
    z[bead_index] = random.randint(-int(lz/2), int(lz/2))

    current_location = None
    for i in range(1, mol_length):
        possible_directions = list(range(1,7))
        if current_location is not None:
            unavailable_direction = reversible_pairs[current_location]
            possible_directions.remove(unavailable_direction)
        
        val = random.choice(possible_directions)
        dx, dy, dz = directions[val]
        x[bead_index+i] = x[bead_index+i-1] + dx 
        y[bead_index+i] = y[bead_index+i-1] + dy
        z[bead_index+i] = z[bead_index+i-1] + dz 

        current_location = val
    bead_index += mol_length

# Apply boundary conditions
for i in range(len(x)):
    if x[i] > lx/2:
        x[i] = lx - x[i]
    elif x[i] < -lx/2:
        x[i] = -lx - x[i]
    else:
        x[i] = x[i]

for i in range(len(y)):
    if y[i] > ly/2:
        y[i] = ly - y[i]
    elif y[i] < -ly/2:
        y[i] = -ly - y[i]
    else:
        y[i] = y[i]

for i in range(len(z)):
    if z[i] > lz/2:
        z[i] = lz - z[i]
    elif z[i] < -lz/2:
        z[i] = -lz - z[i]
    else:
        z[i] = z[i]


# --------------------- randomly select half of chains to crosslink --------------
selected_mols = list(range(total_mols))[:n_chain]
print(selected_mols)
#selected_mols = sample(list(range(total_mols)), total_mols//2)
bead_types = np.ones(total_beads, dtype=int)
type2_beads = []
type3_beads = []

bead_index = 0
for mol_id in range(total_mols):
    mol_length = mol_lengths[mol_id]
    first_bead_id = bead_index
    last_bead_id = bead_index + mol_length - 1
    
    if mol_id in selected_mols:
        # Assign first and last as type 2
        bead_types[first_bead_id] = 2
        bead_types[last_bead_id] = 2
        type2_beads.extend([first_bead_id, last_bead_id])
    else:
        bead_types[first_bead_id] = 3
        bead_types[last_bead_id] = 3
        type3_beads.extend([first_bead_id, last_bead_id])
    
    bead_index += mol_length

# find chain and position of beads on the chain
def get_chain_info(bead_id):
    running_total = 0
    for mol_id, mol_length in enumerate(mol_lengths):
        if bead_id < running_total + mol_length:
            position = bead_id - running_total
            return mol_id, position, mol_length
        running_total += mol_length
    return None


# --------------------- crosslink by density ---------------
def crosslink_by_density(
        type2_beads, type3_beads, x, y, z,
        reaction_extent, min_length, max_length, max_t2_bonds, max_t3_bonds
):
    # basic output
    print("=== CROSSLINK ANALYSIS ===")   
    print(f"Target extent of reaction: {reaction_extent}")
    print(f"Bonding distance range: {min_length} - {max_length} angstroms") 

    # calculate crosslinks
    total_functional_bonds = 0.5*(f_chain*n_chain+f_agent*n_agent)
    target_reacted_bonds = total_functional_bonds * reaction_extent

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
                mol1_id = mol_ids[type2_beads[i]] - 1
                mol2_id = mol_ids[type3_beads[j]] - 1
                if mol1_id != mol2_id:
                    possible_bonds.append((dist, type2_beads[i], type3_beads[j]))
    print("\n=== Total number of possible bonds ===")
    print(f"Found {len(possible_bonds)} possible crosslinking bonds within length constraint")
    if len(possible_bonds) == 0:
        print("WARNING: No possible bonds found in the box to crosslink")
        return [], set()
    
    # add crosslinks
    possible_bonds.sort(key=lambda x: x[0])
    crosslinks = []
    bond_count = {}
    crosslinked_beads = set()
    bead_mol_connections = {}

    for dist, bead1, bead2 in possible_bonds:
        if len(crosslinks) >= target_reacted_bonds:
            break
        if bond_count.get(bead1, 0) >= max_t2_bonds:
            continue
        if bond_count.get(bead2, 0) >= max_t3_bonds:
            continue
        if (bead1, bead2) in crosslinks or (bead2, bead1) in crosslinks:
            continue
        mol1_id = mol_ids[bead1] - 1
        mol2_id = mol_ids[bead2] - 1

        crosslinks.append((bead1, bead2))
        bond_count[bead1] = bond_count.get(bead1, 0) + 1
        bond_count[bead2] = bond_count.get(bead2, 0) + 1
        crosslinked_beads.add(bead1)
        crosslinked_beads.add(bead2)

        if bead1 not in bead_mol_connections:
            bead_mol_connections[bead1] = set()
        if bead2 not in bead_mol_connections:
            bead_mol_connections[bead2] = set()
        bead_mol_connections[bead1].add(mol2_id)
        bead_mol_connections[bead2].add(mol1_id)

    actual_reaction_extent = len(crosslinks) / total_functional_bonds
    print("\n=== ACTUAL CROSSLINK INFORMATION ===")
    print(f"Actual crosslink density: {actual_reaction_extent:.3f}")
    print(f"Crosslinked bonds: {len(crosslinks)} out of {total_functional_bonds} possible bonds have been formed")

    # show bond distribution 
    bond_distribution = {}
    for count in bond_count.values():
        bond_distribution[count] = bond_distribution.get(count, 0) + 1

    print(f"Bond distribution per bead")
    for bonds, beads in sorted(bond_distribution.items()):
        print(f"{bonds} bonds: {beads} beads")

    # end of function
    return crosslinks, crosslinked_beads


# --------------------- generate crosslinks ------------------
crosslinks, crosslinked_beads = crosslink_by_density(type2_beads, type3_beads, x, y, z, reaction_extent, 
                                  min_bond_length, max_bond_length, max_t2_bonds, max_t3_bonds)

# Helper function to get neighbors of a bead in its chain
def get_chain_neighbors(bead_id):
    mol_id, pos, mol_length = get_chain_info(bead_id)
    mol_start = mol_start_indices[mol_id]
    
    neighbors = []
    if pos > 0:  # has previous neighbor
        neighbors.append(mol_start + pos - 1)
    if pos < mol_length - 1:  # has next neighbor
        neighbors.append(mol_start + pos + 1)
    
    return neighbors

# count bonds
chain_bonds = sum(length-1 for length in mol_lengths)
crosslink_bonds = len(crosslinks)
total_bonds = chain_bonds + crosslink_bonds

# Generate and count angles
chain_angles_list = []
bead_id = 0
for mol_id in range(total_mols):
    mol_length = mol_lengths[mol_id]
    for j in range(mol_length-2):
        chain_angles_list.append((bead_id+j, bead_id+j+1, bead_id+j+2))
    bead_id += mol_length

crosslink_angles_list = []
for bead1, bead2 in crosslinks:
    neighbors1 = get_chain_neighbors(bead1)
    neighbors2 = get_chain_neighbors(bead2)
    
    # Create angles: neighbor1 - bead1 - bead2
    for n1 in neighbors1:
        crosslink_angles_list.append((n1, bead1, bead2))
    
    # Create angles: bead1 - bead2 - neighbor2
    for n2 in neighbors2:
        crosslink_angles_list.append((bead1, bead2, n2))

total_angles = len(chain_angles_list) + len(crosslink_angles_list)

# Generate and count dihedrals
chain_dihedrals_list = []
bead_id = 0
for mol_id in range(total_mols):
    mol_length = mol_lengths[mol_id]
    for j in range(mol_length-3):
        chain_dihedrals_list.append((bead_id+j, bead_id+j+1, bead_id+j+2, bead_id+j+3))
    bead_id += mol_length

crosslink_dihedrals_list = []
for bead1, bead2 in crosslinks:
    neighbors1 = get_chain_neighbors(bead1)
    neighbors2 = get_chain_neighbors(bead2)
    
    # For each neighbor of bead1, check if it has a second neighbor
    mol1_id, pos1, len1 = get_chain_info(bead1)
    mol1_start = mol_start_indices[mol1_id]
    
    for n1 in neighbors1:
        # Find the second neighbor of n1 (that's not bead1)
        n1_neighbors = get_chain_neighbors(n1)
        for n1_second in n1_neighbors:
            if n1_second != bead1:
                # Create dihedral: n1_second - n1 - bead1 - bead2
                crosslink_dihedrals_list.append((n1_second, n1, bead1, bead2))
    
    # For each neighbor of bead2, check if it has a second neighbor
    mol2_id, pos2, len2 = get_chain_info(bead2)
    mol2_start = mol_start_indices[mol2_id]
    
    for n2 in neighbors2:
        # Find the second neighbor of n2 (that's not bead2)
        n2_neighbors = get_chain_neighbors(n2)
        for n2_second in n2_neighbors:
            if n2_second != bead2:
                # Create dihedral: bead1 - bead2 - n2 - n2_second
                crosslink_dihedrals_list.append((bead1, bead2, n2, n2_second))

total_dihedrals = len(chain_dihedrals_list) + len(crosslink_dihedrals_list)

print(f"\n=== TOPOLOGY SUMMARY ===")
print(f"Total bonds: {total_bonds} (chain: {chain_bonds}, crosslink: {crosslink_bonds})")
print(f"Total angles: {total_angles} (chain: {len(chain_angles_list)}, crosslink: {len(crosslink_angles_list)})")
print(f"Total dihedrals: {total_dihedrals} (chain: {len(chain_dihedrals_list)}, crosslink: {len(crosslink_dihedrals_list)})")


# --------------------- write data file --------------
dist_str = "_".join(f"{length}x{count}" for length, count in sorted(mol_length_distribution.items()))
filename = f"polydisperse_{dist_str}_crosslink_{reaction_extent:.2f}.data"

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
    LAMMPS.write("{} {} # Type1\n".format(1, mass))
    LAMMPS.write("{} {} # Type2\n".format(2, mass))
    LAMMPS.write("{} {} # Type3_crosslinker\n\n".format(3, mass))

    # atom section 
    LAMMPS.write("Atoms #full\n\n")
    for bead in range(total_beads):
        LAMMPS.write("{} {} {} {} {} {} {}\n".format(bead+1, mol_ids[bead], bead_types[bead], 0, x[bead], y[bead], z[bead]))

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
    # chain angles
    for a1, a2, a3 in chain_angles_list:
        LAMMPS.write("{} {} {} {} {}\n".format(angle_id, 1, a1+1, a2+1, a3+1))
        angle_id += 1
    # crosslink angles
    for a1, a2, a3 in crosslink_angles_list:
        LAMMPS.write("{} {} {} {} {}\n".format(angle_id, 1, a1+1, a2+1, a3+1))
        angle_id += 1

    # dihedral section 
    LAMMPS.write("\nDihedrals\n\n")
    dihedral_id = 1
    # chain dihedrals
    for d1, d2, d3, d4 in chain_dihedrals_list:
        LAMMPS.write("{} {} {} {} {} {}\n".format(dihedral_id, 1, d1+1, d2+1, d3+1, d4+1))
        dihedral_id += 1
    # crosslink dihedrals
    for d1, d2, d3, d4 in crosslink_dihedrals_list:
        LAMMPS.write("{} {} {} {} {} {}\n".format(dihedral_id, 1, d1+1, d2+1, d3+1, d4+1))
        dihedral_id += 1

print(f"\n=== FILE WRITTEN ===")
print(f"Data file written to: {filename}")
