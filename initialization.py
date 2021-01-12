import numpy as np
from ovito.io import *
from ovito.modifiers import CreateBondsModifier, CoordinationAnalysisModifier, CalculateDisplacementsModifier

def initialize_system(filename):
    # Obtain the pipeline, the data, the number of atoms, of frames, of different types of atoms of the simulation
    pipeline = import_file(filename)
    data = pipeline.compute()
    num_of_atoms = data.particles.count  # Get number of atoms
    num_of_frames = pipeline.source.num_frames  # Get number of frames
    num_of_type = np.unique(data.particles.particle_types).shape[0]  # Get number of different types of atoms
    return [pipeline, data, num_of_atoms, num_of_frames, num_of_type]


def set_bond_length(bond_lengths, pipeline, num_of_type):
    # Define the bond length criterion for each type of bond
    cbm = CreateBondsModifier(mode=CreateBondsModifier.Mode.Pairwise)
    bond_type = 0
    for atom_type_1 in range(1, num_of_type + 1):
        for atom_type_2 in range(atom_type_1, num_of_type + 1):
            cbm.set_pairwise_cutoff(atom_type_1, atom_type_2, bond_lengths[bond_type])
            bond_type += 1
    pipeline.modifiers.append(cbm)

def get_atom_types(data, num_of_atoms):
    atom_types = np.zeros([num_of_atoms], dtype=int)
    atom_types[data.particles['Particle Identifier']-1] = data.particles.particle_types[:]
    return atom_types

