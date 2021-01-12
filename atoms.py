import numpy as np

# Class that defines the description of the atoms for the atomic-level description

class Atoms:
    def __init__(self, num_of_type, atom_types):
        self.num_of_type = num_of_type
        self.num_atom_descriptors = 1 + num_of_type  # One atom is described by its type and the type of its neighbors
        self.atom_types = atom_types

    def get_atom_features_for_reaction(self, reaction_info, real_bonds):
        # Compute the atom descriptions of 2 atoms involved in a bond change. real_bonds need to be symmetric.
        atom_description_1 = self.get_atom_features(reaction_info[0], real_bonds)
        atom_description_2 = self.get_atom_features(reaction_info[1], real_bonds)
        m = np.argwhere(atom_description_1[0] - atom_description_2[0] != 0)
        # Order the descriptions of the atoms by lexicographic order.
        if m.shape[0] == 0 or atom_description_1[0][min(m)] < atom_description_2[0][min(m)]:
            atoms_description = np.concatenate((atom_description_1[0], atom_description_2[0]))
        else:
            atoms_description = np.concatenate((atom_description_2[0], atom_description_1[0]))
        return atoms_description

    def get_atom_features(self, atom, real_bonds):
        # Compute the atom description of "atom". real_bonds need to be symmetric.
        atom_description = np.zeros([1, self.num_atom_descriptors])
        atom_description[0][0] = self.atom_types[atom]
        atom_description[0][1:(self.num_of_type + 1)] = self.get_type_first_nn(atom, real_bonds)
        return atom_description

    def get_type_first_nn(self, atom, real_bonds):
        # Obtain the type (C,H,O,...) of the atoms that are bonded with "atom". real_bonds need to be symmetric.
        bonded_atom = np.where(real_bonds[atom, :])[0]
        type_first_nn = np.zeros([self.num_of_type])
        for i in bonded_atom:
            type_first_nn[self.atom_types[i]-1] += 1
        return type_first_nn

    def get_all_atoms_features(self, real_bonds):
        # Compute the atom descriptions of all the atoms in the system. real_bonds need to be symmetric.
        all_atoms_features = np.zeros([real_bonds.shape[0], self.num_atom_descriptors], dtype=int)
        for atom in range(real_bonds.shape[0]):
            all_atoms_features[atom] = self.get_atom_features(atom, real_bonds)
        return all_atoms_features

    def update_all_atoms_features(self, real_bonds, bond_change_frame, all_atoms_features):
        # Update the atom descriptions after a set of reactions "bond_change_frame" happened. real_bonds need
        # to be symmetric.
        atoms_involved = np.unique(bond_change_frame[:, 0:2])
        for atom in atoms_involved:
            all_atoms_features[atom] = self.get_atom_features(atom, real_bonds)

