# ---------------------------------import modules and packages ----------------------------------
import numpy as np
import random

import MDAnalysis as mda

from scipy.spatial.transform import Rotation
from numpy.linalg import norm
from doepy import build



# ------------------------------------ define the frame of the new model ------------------
# define number of replicated chains
n= 10

print(f"Chain number = {n}")


# read data files
u = mda.Universe(r"./pe.data", atom_style='id resid type charge x y z')
filename = r"./pe1.data"


# read and redefine atoms for the new model
atoms = len(u.atoms) # number of atoms of the current chain
density = 0.1 # density of the multiple chain model
total_mass = sum(u.atoms.masses) * n # mass of the new model

lx = ly = lz = (total_mass/density)**(1/3) # edge of the box
lo = 0.05
hi = 0.95



# ------------------------------------ extract information of the initial model from the beginning data file --------------------
# define functions to extract information from initial data file and make extension
# extract Atoms
def extract_atoms(filename):
    with open(filename, 'r') as file:
        atom = []
        for line in file:
            if "Atoms" in line:
                break
        for line in file:
            if "Bonds" in line:
                break
            data = line.strip().split()
            atom.append(data)
        return atom

# extract Bonds
def extract_bonds(filename):
    with open(filename, 'r') as file:
        bonds = []
        for line in file:
            if "Bonds" in line:
                break
        for line in file:
            if "Angles" in line:
                break
            data = line.strip().split()
            bonds.append(data)
        return bonds

# extract Angles
def extract_angles(filename):
    with open(filename, 'r') as file:
        angles = []
        for line in file:
            if "Angles" in line:
                break
        for line in file:
            if "Dihedrals" in line:
                break
            data = line.strip().split()
            angles.append(data)
        return angles

# extract Dihedrals
def extract_dihedrals(filename):
    with open(filename, 'r') as file:
        dihedrals = []
        for line in file:
            if "Dihedrals" in line:
                break
        for line in file:
            if "Impropers" in line:
                break
            data = line.strip().split()
            dihedrals.append(data)
        return dihedrals

# extract Impropers
def extract_impropers(filename):
    with open(filename, "r") as file:
        impropers = []
        for line in file:
            if "Impropers" in line:
                break
        for line in file:
            if "Impropers1" in line:
                break
            data = line.strip().split()
            impropers.append(data)
        return impropers



# ------------------------------------ generate random points in the new simulation box -------------------
# produce n random points in the box
pos_target = u.atoms.positions

data = build.maximin(
    {'x': [lx*lo,lx*hi],
     'y':[ly*lo, ly*hi],
     'z':[lz*lo, lz*hi],},
    num_samples= n
)

x = data.iloc[:,0].values
y = data.iloc[:,1].values
z = data.iloc[:,2].values

randomseeds = np.vstack((x, y, z)).T[:n]



# -------------------------------- save data collected from the old data file -------------------------------
# lists for mol & moltype
mol = []
atomtype = []

# bond & bondtype
bondtype = []
bonda = []
bondb = []

# angle & angletype
angletype = []
anglea = []
angleb = []
anglec = []

# dihedral & dihedraltype
dihedraltype = []
dihedrala = []
dihedralb = []
dihedralc = []
dihedrald = []

# improper & impropertype
impropertype = []
impropera = []
improperb = []
improperc = []
improperd = []


# define a zero matrix to save 3D spatial coordinates of all atoms of n chains
pos = np.zeros((n,len(pos_target),3))

# extract data from the initial data file and replicate the data
for m in range(n):
    # extract atom type and molecular type from initial data file
    atom = [[float(x) for x in row] for row in extract_atoms(filename)]
    atom = [matrix for matrix in atom if matrix]
    atomtype.append(np.array(atom)[:,2])
    str(atomtype).replace("'", "")
    mol.append(np.array(atom)[:,1]+m)
    str(mol).replace("'", "")

    # extract & save bond information
    bonds = [[int(x) for x in row] for row in extract_bonds(filename)]
    bonds = [matrix for matrix in bonds if matrix]
    bondtype.append(np.array(bonds)[:,1])
    bonda.append(np.array(bonds)[:,2] + len(u.atoms)*m)
    bondb.append(np.array(bonds)[:,3] + len(u.atoms)*m)

    # extract & save angle information
    angles = [[int(x) for x in row] for row in extract_angles(filename)]
    angles = [matrix for matrix in angles if matrix]
    angletype.append(np.array(angles)[:,1])
    anglea.append(np.array(angles)[:,2] + len(u.atoms)*m)
    angleb.append(np.array(angles)[:,3] + len(u.atoms)*m)
    anglec.append(np.array(angles)[:,4] + len(u.atoms)*m)

    # extract & save dihedral information
    dihedrals = [[int(x) for x in row] for row in extract_dihedrals(filename)]
    dihedrals = [matrix for matrix in dihedrals if matrix]
    dihedraltype.append(np.array(dihedrals)[:,1])
    dihedrala.append(np.array(dihedrals)[:,2] + len(u.atoms)*m)
    dihedralb.append(np.array(dihedrals)[:,3] + len(u.atoms)*m)
    dihedralc.append(np.array(dihedrals)[:,4] + len(u.atoms)*m)
    dihedrald.append(np.array(dihedrals)[:,5] + len(u.atoms)*m)
    '''
    # extract & save improper information
    impropers = [[int(x) for x in row] for row in extract_impropers(filename)]
    impropers = [matrix for matrix in impropers if matrix]
    impropertype.append(np.array(impropers)[:,1])
    impropera.append(np.array(impropers)[:,2] + len(u.atoms)*m)
    improperb.append(np.array(impropers)[:,3] + len(u.atoms)*m)
    improperc.append(np.array(impropers)[:,4] + len(u.atoms)*m)
    improperd.append(np.array(impropers)[:,5] + len(u.atoms)*m)
    '''

    # translation and rotations of the atoms in the polymer chain
    pos_target = u.atoms.positions - [max(u.atoms.positions[:,0]), max(u.atoms.positions[:,1]), max(u.atoms.positions[:,2])]

    axis = np.random.randint(-15,15,(1,3)).tolist()
    axis = axis / norm(axis)
    theta = random.uniform(0,2*np.pi)
    initial_x = randomseeds[m][0]
    initial_y = randomseeds[m][1]
    initial_z = randomseeds[m][2]

    for j in range(len(pos_target)):
        rotation = Rotation.from_rotvec(theta * axis)
        pos[m][j][0] = rotation.apply(pos_target)[j][0] + initial_x
        pos[m][j][1] = rotation.apply(pos_target)[j][1] + initial_y
        pos[m][j][2] = rotation.apply(pos_target)[j][2] + initial_z


# --------------------------- write LAMMPS data files ---------
with open(r"./chain{}.data".format(n), "w")as LAMMPS:
    # write the comment line at the beginning
    LAMMPS.write("The random polymer chain system from python\n\n")

    # Header Line
    LAMMPS.write('{} atoms\n'.format(len(pos[0])*n))
    LAMMPS.write('{} bonds\n'.format(len(u.bonds)*n))
    LAMMPS.write('{} angles\n'.format(len(u.angles)*n))
    LAMMPS.write('{} dihedrals\n'.format(len(u.dihedrals)*n))
    LAMMPS.write('{} impropers\n\n'.format(len(u.impropers)*n))

    # Type Definition
    LAMMPS.write('{} atom types\n'.format(2))
    LAMMPS.write('{} bond types\n'.format(2))
    LAMMPS.write('{} angle types\n'.format(1))
    LAMMPS.write('{} dihedral types\n'.format(1))
    #LAMMPS.write('{} improper types\n\n'.format(0))

    # Size of the Box
    LAMMPS.write('{} {} xlo xhi\n'.format(0, lx))
    LAMMPS.write('{} {} ylo yhi\n'.format(0, ly))
    LAMMPS.write('{} {} zlo zhi\n'.format(0, lz))

    # Masses of Atoms
    LAMMPS.write('\nMasses\n\n')

    LAMMPS.write('{} {}\n'.format(1, 1.008))
    LAMMPS.write('{} {}\n'.format(2, 12.011))


    # Specify Atoms
    # Full-Atom: atom-id; molecule-id; atom-type; charge; x; y; z
    LAMMPS.write('\n Atoms # full \n\n')
    id = 1
    for i in range(len(pos)):
        for j in range(len(pos[i])):
            LAMMPS.write('{} {} {} {} {:.2f} {:.2f} {:.2f}\n'.format(id, int(mol[i][j]), int(atomtype[i][j]), 0,
                                                                   pos[i][j][0],
                                                                   pos[i][j][1],
                                                                   pos[i][j][2],)) 
            id = id +1

    # Specify bonds
    # Bonds: bond-id; bondtype; atom-a; atom-b
    LAMMPS.write('\nBonds   \n\n')
    id = 1
    for i in range(len(bonda)):
        for j in range(len(bonda[0])):
            LAMMPS.write('{} {} {} {} \n'.format(id, bondtype[i][j], bonda[i][j], bondb[i][j]))
            id = id + 1

    # Specify angles
    # Angles: angle-id; angletype; atom-a; atom-b; atom-c
    LAMMPS.write('\nAngles   \n\n')
    id = 1
    for i in range(len(anglea)):
        for j in range(len(anglea[0])):
            LAMMPS.write('{} {} {} {} {}\n'.format(id, angletype[i][j], anglea[i][j], angleb[i][j], anglec[i][j]))
            id = id + 1

    # Specify dihedrals
    # Dihedrals: dihedral-id; dihedraltype; atom-a; atom-b; atom-c; atom-d
    LAMMPS.write('\nDihedrals   \n\n')
    id = 1
    for i in range(len(dihedrala)):
        for j in range(len(dihedrala[0])):
            LAMMPS.write('{} {} {} {} {} {}\n'.format(id, dihedraltype[i][j], dihedrala[i][j], dihedralb[i][j], dihedralc[i][j], dihedrald[i][j]))
            id = id + 1
    '''
    # Specify impropers
    # Impropers: improper-id; impropertype; atom-a; atom-b; atom-c; atom-c
    LAMMPS.write('\nImpropers   \n\n')
    id = 1
    for i in range(len(impropera)):
        for j in range(len(impropera[0])):
            LAMMPS.write('{} {} {} {} {} {}\n'.format(id, impropertype[i][j], impropera[i][j], improperb[i][j], improperc[i][j], improperd[i][j]))
            id = id + 1
    '''
