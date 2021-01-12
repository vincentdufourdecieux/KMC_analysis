import numpy as np
import time

# Class that creates the reaction list, store their descriptions, the number of times each reaction occurred,
# the number of times each reaction could have occurred.

class Reactions:
    def __init__(self, num_of_type, reaction_dict, reaction_type, atoms, MD_or_KMC, molecules):
        self.num_of_type = num_of_type
        self.reaction_occurrences = np.array([])  # Number of times a reaction happen
        self.reaction_occurrences_per_frame = {}
        self.reaction_dict = reaction_dict  # All the reactions that are in memory
        self.reaction_type = reaction_type  # Type of reaction, simplify some calculations.
        self.num_atom_descriptors = 1 + num_of_type
        self.num_reaction_descriptors = 1
        self.num_descriptors = 2 * self.num_atom_descriptors + self.num_reaction_descriptors
        self.atoms = atoms
        self.MD_or_KMC = MD_or_KMC  # 0 for MD, 1 for KMC
        if np.shape(self.reaction_dict)[0] == 0:
            self.reaction_dict.shape = [0, self.num_descriptors]
        self.all_atoms_features = np.array([])
        self.num_of_reactants = np.zeros([self.reaction_dict.shape[0], 2], dtype=int)
        self.atom_for_reactions = {}
        self.molecules = molecules
        self.step_with_reax = 0

    def update_reactions_and_real_bonds(self, reaction_frame, real_bonds, frame,
                                        molecules, old_molecules_frame, old_molecule_bond_change, new_molecules_frame,
                                        new_molecule_bond_change):
        # Check the reactions that happened during the frame and get their description, if they happened before just
        # add an occurence, otherwise add the reaction description to the list of reaction, their type and count an
        # occurrence.
        for reax in range(reaction_frame.shape[0]):
            real_bonds_symmetric_array = real_bonds.toarray()
            real_bonds_symmetric_array = real_bonds_symmetric_array + real_bonds_symmetric_array.T
            reax_description = self.get_reaction_description(reaction_frame[reax, :], real_bonds_symmetric_array)
            if not np.equal(reax_description, self.reaction_dict).all(axis=1).any():
                self.reaction_dict = np.append(self.reaction_dict, reax_description, 0)
                self.reaction_type = np.append(self.reaction_type, self.get_reaction_type(reax_description))
                self.reaction_occurrences = np.append(self.reaction_occurrences, 1)
                self.reaction_occurrences_per_frame[len(self.reaction_dict) - 1] = [frame]
                reaction_frame[reax][3] = self.reaction_occurrences.shape[0] - 1
            else:
                idx = np.argwhere(np.equal(reax_description, self.reaction_dict).all(axis=1))[0][0]
                self.reaction_occurrences[idx] += 1
                self.reaction_occurrences_per_frame[idx].append(frame)
                reaction_frame[reax][3] = idx
            real_bonds[reaction_frame[reax, 0], reaction_frame[reax, 1]] = reaction_frame[reax, 2]

    def get_reaction_description(self, reax, real_bonds):
        # Get the description of the reaction which is in order:
        # - 0 if the bond between the atoms is broken, 1 if it is created
        # - Description of the first atom
        # - Description of the second atom
        reax_description = np.zeros([1, self.num_descriptors])
        reax_description[0][0] = reax[2]
        atom_description = self.atoms.get_atom_features_for_reaction(reax, real_bonds)
        reax_description[0, self.num_reaction_descriptors:
                            (2 * self.num_atom_descriptors + self.num_reaction_descriptors)] = atom_description
        return reax_description

    def get_reaction_type(self, reax_description):
        # If a reaction is the breakage of a bond, its type is 0. If it is the creation of a bond between 2 different
        # atoms it is 1, and if it is hte creation of a bond between 2 atoms that are the same it is 2. The reaction
        # type helps when counting the number of times the reaction could have happened.
        if reax_description[0, 0] == 0:
            return 0
        else:
            if np.equal(reax_description[0,
                        self.num_reaction_descriptors:self.num_atom_descriptors + self.num_reaction_descriptors],
                        reax_description[0, self.num_atom_descriptors + self.num_reaction_descriptors:
                        2 * self.num_atom_descriptors + self.num_reaction_descriptors]).all():
                return 2
            else:
                return 1

    def get_h(self, first_frame, bond_change, start_frame, end_frame):
        # Get the number of times each reaction could have happened (h) starting from the state of the system defined
        # by "first_frame" at the frame "start_frame" until the "end_frame".
        h_tot = np.zeros([self.reaction_dict.shape[0]], dtype=int)
        h_per_frame = np.zeros([len(self.reaction_dict), end_frame - start_frame])
        self.num_of_reactants = np.zeros([self.reaction_dict.shape[0], 2], dtype=int)
        bond_matrix = first_frame.copy()
        self.all_atoms_features = self.atoms.get_all_atoms_features(bond_matrix)
        h_prev = self.get_h_frame(bond_matrix)
        h_per_frame[:, 0] = h_prev
        for frame in range(start_frame, end_frame):
            if frame % 1000 == 0:
                print(frame, flush=True)
                print(time.process_time(), flush=True)
            h_tot += h_prev
            h_per_frame[:, frame - start_frame] = h_prev
            if frame in bond_change.keys():
                # Update h after the reaction happened
                h_tot += self.get_correction(bond_change[frame][:, 3])
                h_per_frame[:, frame - start_frame] += self.get_correction(bond_change[frame][:, 3])
                h_prev = self.update_all_after_reax(h_prev, bond_matrix, bond_change[frame])
        return [h_tot, h_per_frame]

    def get_h_frame(self, bond_matrix):
        # We get the number of times h each reaction could have happened for one frame.
        if self.all_atoms_features.shape[0] == 0:
            self.all_atoms_features = self.atoms.get_all_atoms_features(bond_matrix)
        h = np.zeros([self.reaction_dict.shape[0]], dtype=int)
        if self.MD_or_KMC == 1 and self.step_with_reax == 0:
            self.molecules.initialize_molecules(bond_matrix, 0)
            for i in range(self.reaction_dict.shape[0]):
                self.atom_for_reactions[i] = np.array([], dtype=int)
                self.atom_for_reactions[i].shape = [0, 2]
        for reax in range(self.reaction_dict.shape[0]):
            if self.MD_or_KMC == 0:
                if self.reaction_type[reax] == 2:
                    h[reax] = self.get_h_creation_same_atoms(bond_matrix, reax)
                elif self.reaction_type[reax] == 1:
                    h[reax] = self.get_h_creation_different_atoms(bond_matrix, reax)
                else:
                    h[reax] = self.get_h_breakage(bond_matrix, reax)
            elif self.MD_or_KMC == 1:
                if self.reaction_type[reax] == 2:
                    h[reax], self.atom_for_reactions[reax] = self.get_h_creation_same_atoms(bond_matrix, reax)
                elif self.reaction_type[reax] == 1:
                    h[reax], self.atom_for_reactions[reax] = self.get_h_creation_different_atoms(bond_matrix, reax)
                else:
                    h[reax], self.atom_for_reactions[reax] = self.get_h_breakage(bond_matrix, reax)
        return h

    def get_h_creation_same_atoms(self, bond_matrix, reax):
        # Get the number of times the reaction "reax" could have happened. "reax" must be a reaction that is the
        # creation of a bond between 2 atoms with the same descriptions.
        reactant = self.reaction_dict[reax][self.num_reaction_descriptors:
                                            (self.num_atom_descriptors + self.num_reaction_descriptors)]
        atom_counted = np.where((self.all_atoms_features == reactant).all(-1))[0]
        num_of_reactants_temp = atom_counted.shape[0]
        if self.MD_or_KMC == 1:
            atom_for_reaction = np.array([], dtype=int)
            atom_for_reaction.shape = [0, 2]
        h = num_of_reactants_temp * (num_of_reactants_temp - 1) / 2
        # Atoms already bonded can't bond again
        bonds_between_reactants = bond_matrix[atom_counted, :]
        bonds_between_reactants = bonds_between_reactants[:, atom_counted]
        h -= np.count_nonzero(bonds_between_reactants) // 2
        if self.MD_or_KMC == 1 and h > 0:
            atom_for_reaction = np.where(bonds_between_reactants == 0)
            atom_for_reaction = np.vstack((atom_counted[atom_for_reaction[0]], atom_counted[atom_for_reaction[1]])).T
            atom_for_reaction = np.delete(atom_for_reaction,
                                          np.where(atom_for_reaction[:, 0] == atom_for_reaction[:, 1])[0], axis=0)
            atom_for_reaction = np.sort(atom_for_reaction, axis=1)
            atom_for_reaction = np.unique(atom_for_reaction, axis=0)
        self.num_of_reactants[reax] = [num_of_reactants_temp, 0]
        if self.MD_or_KMC == 0:
            return h
        if self.MD_or_KMC == 1:
            return h, atom_for_reaction

    def get_h_creation_different_atoms(self, bond_matrix, reax):
        # Get the number of times the reaction "reax" could have happened. "reax" must be a reaction that is the
        # creation of a bond between 2 atoms with different descriptions.
        reactant_1 = self.reaction_dict[reax][self.num_reaction_descriptors:
                                              (self.num_atom_descriptors + self.num_reaction_descriptors)]
        reactant_2 = self.reaction_dict[reax][(self.num_atom_descriptors + self.num_reaction_descriptors):
                                              2 * (self.num_atom_descriptors + self.num_reaction_descriptors)]
        atom_counted_1 = np.where((self.all_atoms_features == reactant_1).all(-1))[0]
        atom_counted_2 = np.where((self.all_atoms_features == reactant_2).all(-1))[0]
        if self.MD_or_KMC == 1:
            atom_for_reaction = np.array([], dtype=int)
            atom_for_reaction.shape = [0, 2]
        num_of_reactants_1 = atom_counted_1.shape[0]
        num_of_reactants_2 = atom_counted_2.shape[0]
        h = num_of_reactants_1 * num_of_reactants_2
        # Atoms already bonded can't bond again
        bonds_between_reactants = bond_matrix[atom_counted_1, :]
        bonds_between_reactants = bonds_between_reactants[:, atom_counted_2]
        h -= np.count_nonzero(bonds_between_reactants)
        if self.MD_or_KMC == 1 and h > 0:
            atom_for_reaction = np.where(bonds_between_reactants == 0)
            atom_for_reaction = np.vstack(
                (atom_counted_1[atom_for_reaction[0]], atom_counted_2[atom_for_reaction[1]])).T
        self.num_of_reactants[reax] = [num_of_reactants_1, num_of_reactants_2]
        if self.MD_or_KMC == 0:
            return h
        if self.MD_or_KMC == 1:
            return h, atom_for_reaction

    def get_h_breakage(self, bond_matrix, reax):
        # Get the number of times the reaction "reax" could have happened. "reax" must be a reaction that is the
        # breakage of a bond between 2 atoms with different descriptions.
        reactant_1 = self.reaction_dict[reax][self.num_reaction_descriptors:
                                              (self.num_atom_descriptors + self.num_reaction_descriptors)]
        reactant_2 = self.reaction_dict[reax][(self.num_atom_descriptors + self.num_reaction_descriptors):
                                              2 * (self.num_atom_descriptors + self.num_reaction_descriptors)]
        atom_counted_1 = np.where((self.all_atoms_features == reactant_1).all(-1))[0]
        atom_counted_2 = np.where((self.all_atoms_features == reactant_2).all(-1))[0]
        if self.MD_or_KMC == 1:
            atom_for_reaction = np.array([], dtype=int)
            atom_for_reaction.shape = [0, 2]
        bonds_between_reactants = bond_matrix[atom_counted_1, :]
        bonds_between_reactants = bonds_between_reactants[:, atom_counted_2]
        h = np.count_nonzero(bonds_between_reactants)
        if self.MD_or_KMC == 1 and h > 0:
            atom_for_reaction = np.where(bonds_between_reactants == 1)
            atom_for_reaction = np.vstack(
                (atom_counted_1[atom_for_reaction[0]], atom_counted_2[atom_for_reaction[1]])).T
        if np.equal(reactant_1, reactant_2).all():
            h = h // 2
            if self.MD_or_KMC == 1:
                atom_for_reaction = np.delete(atom_for_reaction,
                                              np.where(atom_for_reaction[:, 0] > atom_for_reaction[:, 1]), axis=0)
        self.num_of_reactants[reax] = [h, 0]
        if self.MD_or_KMC == 0:
            return h
        if self.MD_or_KMC == 1:
            return h, atom_for_reaction

    def get_correction(self, reax_occur_at_frame):
        # Calculate the correction if a reaction happens when not ready.
        correction_frame = np.zeros([self.reaction_dict.shape[0]], dtype=int)
        reax, counts = np.unique(reax_occur_at_frame, return_counts=True)
        for idx in range(reax.shape[0]):
            if self.reaction_type[reax[idx]] == 0:
                if counts[idx] > self.num_of_reactants[reax[idx]][0]:
                    correction_frame[reax[idx]] = counts[idx] - self.num_of_reactants[reax[idx]][0]
            elif self.reaction_type[reax[idx]] == 1:
                if counts[idx] > min(self.num_of_reactants[reax[idx]]):
                    correction_frame[reax[idx]] = counts[idx] - min(self.num_of_reactants[reax[idx]])
            else:
                if 2 * counts[idx] > self.num_of_reactants[reax[idx]][0]:
                    correction_frame[reax[idx]] = counts[idx] - (self.num_of_reactants[reax[idx]][0] // 2)
        return correction_frame

    def update_all_after_reax(self, h_prev, bond_matrix, bond_change, frame=-1):
        # Update h, num_of_reactants, bond_matrix (adjacency matrix), all_atoms_features, after some bond changes.
        h = h_prev.copy()
        atoms_involved = np.unique(bond_change[:, :2])
        self.step_with_reax += 1
        if self.MD_or_KMC == 1:
            self.molecules.update_after_reaction(frame, bond_change)
        [h_old, num_of_reactants_old] = self.modif_h(bond_matrix, atoms_involved, 0)
        self.update_bond_matrix(bond_matrix, bond_change)
        self.atoms.update_all_atoms_features(bond_matrix,
                                            bond_change, self.all_atoms_features)  # Update after so we count the
        # reactants before the reactions happen
        self.num_of_reactants -= num_of_reactants_old
        [h_new, num_of_reactants_new] = self.modif_h(bond_matrix, atoms_involved, 1)
        h = h + h_new - h_old
        self.num_of_reactants += num_of_reactants_new
        return h

    def modif_h(self, bond_matrix, atoms_involved, tag):
        # Compute the modifications in h, num_of_reactants, and atom_for_reaction before (tag == 0) or after (tag
        # == 1) some bond changes.
        [modif_h_breakage, modif_num_of_reactants_breakage] = \
            self.modif_h_breakage_full(bond_matrix, atoms_involved, tag)
        [modif_h_creation, modif_num_of_reactants_creation] = \
            self.modif_h_creation_full(bond_matrix, atoms_involved, tag)

        modif_h = modif_h_breakage + modif_h_creation
        modif_num_of_reactants = modif_num_of_reactants_breakage + modif_num_of_reactants_creation

        return [modif_h, modif_num_of_reactants]

    def modif_h_breakage_full(self, bond_matrix, atoms_involved, tag):
        # Compute the modifications in h and num_of_reactants for the reactions that are the breakage of a bond.
        breakage_reactions = np.where(self.reaction_type == 0)[0]
        modif_h = np.zeros([self.reaction_dict.shape[0]], dtype=int)
        modif_num_of_reactants = np.zeros([self.reaction_dict.shape[0], 2], dtype=int)
        bonded_atoms = np.where(bond_matrix[atoms_involved, :])
        bonded_atoms_involved = bond_matrix[atoms_involved, :]
        bonded_atoms_involved = bonded_atoms_involved[:, atoms_involved]
        atoms_as_reactant_1 = np.where(
            (self.all_atoms_features[atoms_involved, None]
             == self.reaction_dict[breakage_reactions,
                self.num_reaction_descriptors:(self.num_atom_descriptors + self.num_reaction_descriptors)]).all(-1))
        atoms_as_reactant_2 = \
            np.where((self.all_atoms_features[atoms_involved, None] ==
                      self.reaction_dict[breakage_reactions,
                      (self.num_atom_descriptors + self.num_reaction_descriptors):
                      2 * (self.num_atom_descriptors +
                           self.num_reaction_descriptors)]).all(-1))
        unique_reax = np.unique(np.concatenate([atoms_as_reactant_1[1], atoms_as_reactant_2[1]]))
        for reax in unique_reax:
            atoms_as_reactant_1_for_reax = atoms_as_reactant_1[0][np.where(atoms_as_reactant_1[1] == reax)]
            atoms_as_reactant_2_for_reax = atoms_as_reactant_2[0][np.where(atoms_as_reactant_2[1] == reax)]
            self.modif_h_breakage(atoms_as_reactant_1_for_reax, atoms_as_reactant_2_for_reax, bonded_atoms,
                                  atoms_involved, modif_num_of_reactants, modif_h, breakage_reactions[reax],
                                  bonded_atoms_involved, tag)
        return [modif_h, modif_num_of_reactants]

    def modif_h_creation_full(self, bond_matrix, atoms_involved, tag):
        # Compute the modifications in h and num_of_reactants for the reactions that are the creation of a bond.
        creation_reactions = np.where(self.reaction_type > 0)[0]
        modif_h = np.zeros([self.reaction_dict.shape[0]], dtype=int)
        modif_num_of_reactants = np.zeros([self.reaction_dict.shape[0], 2], dtype=int)
        bonded_atoms = np.where(bond_matrix[atoms_involved, :])
        bonded_atoms_involved = bond_matrix[atoms_involved, :]
        bonded_atoms_involved = bonded_atoms_involved[:, atoms_involved]
        atoms_as_reactant_1 = np.where(
            (self.all_atoms_features[atoms_involved, None] ==
             self.reaction_dict[creation_reactions,
             self.num_reaction_descriptors:(self.num_atom_descriptors + self.num_reaction_descriptors)]).all(-1))
        atoms_as_reactant_2 = \
            np.where((self.all_atoms_features[atoms_involved, None] ==
                      self.reaction_dict[creation_reactions,
                      (self.num_atom_descriptors + self.num_reaction_descriptors):
                      2 * (self.num_atom_descriptors + self.num_reaction_descriptors)]).all(-1))
        unique_reax = np.unique(np.concatenate([atoms_as_reactant_1[1], atoms_as_reactant_2[1]]))
        for reax in unique_reax:
            atoms_as_reactant_1_for_reax = atoms_as_reactant_1[0][np.where(atoms_as_reactant_1[1] == reax)]
            atoms_as_reactant_2_for_reax = atoms_as_reactant_2[0][np.where(atoms_as_reactant_2[1] == reax)]
            if self.reaction_type[creation_reactions[reax]] == 1:
                self.modif_h_creation_different_atoms(atoms_as_reactant_1_for_reax, atoms_as_reactant_2_for_reax,
                                                      bonded_atoms, atoms_involved, modif_num_of_reactants, modif_h,
                                                      creation_reactions[reax], bonded_atoms_involved, tag)
            elif self.reaction_type[creation_reactions[reax]] == 2:
                self.modif_h_creation_same_atoms(atoms_as_reactant_1_for_reax, bonded_atoms, atoms_involved,
                                                 modif_num_of_reactants, modif_h,
                                                 creation_reactions[reax], bonded_atoms_involved, tag)
        return [modif_h, modif_num_of_reactants]

    def modif_h_breakage(self, atoms_as_reactant_1_for_reax, atoms_as_reactant_2_for_reax, bonded_atoms, atoms_involved,
                         modif_num_of_reactants, modif_h, reax, bonded_atoms_involved, tag):
        # Compute the modifications in h, num_of_reactants, and atom_for_reaction before (tag2 == 0) or after (tag2
        # == 1) for a reaction that break a bond.
        reactant_1 = self.reaction_dict[reax,
                     self.num_reaction_descriptors:(self.num_atom_descriptors + self.num_reaction_descriptors)]
        reactant_2 = self.reaction_dict[reax, (self.num_atom_descriptors + self.num_reaction_descriptors):
                                              2 * (self.num_atom_descriptors + self.num_reaction_descriptors)]
        bonded_atoms_involved_reax = bonded_atoms_involved[atoms_as_reactant_1_for_reax, :]
        bonded_atoms_involved_reax = bonded_atoms_involved_reax[:, atoms_as_reactant_2_for_reax]
        if self.MD_or_KMC == 1:
            pairs_reactant = np.array([], dtype=int)
            pairs_reactant.shape = [0, 2]
        if (reactant_1 == reactant_2).all():
            for atom in atoms_as_reactant_1_for_reax:
                idx_in_bonded_atoms = np.where(bonded_atoms[0] == atom)[0]
                atoms_bonded = bonded_atoms[1][idx_in_bonded_atoms]
                reactant_atoms_bonded = bonded_atoms[1][
                    idx_in_bonded_atoms[np.where((self.all_atoms_features[atoms_bonded] == reactant_2).all(-1))[0]]]
                modif_h[reax] += reactant_atoms_bonded.shape[0]
                if self.MD_or_KMC == 1:
                    pairs_reactant_atom = np.vstack((
                        np.ones(reactant_atoms_bonded.shape[0], dtype=int) * atoms_involved[
                            atom], reactant_atoms_bonded)).T
                    pairs_reactant = np.vstack((pairs_reactant, pairs_reactant_atom))
            modif_h[reax] -= np.count_nonzero(bonded_atoms_involved_reax) // 2
            if self.MD_or_KMC == 1:
                if tag == 0 and pairs_reactant.shape[0] > 0:
                    self.remove_from_atom_for_reaction(-1, 1, pairs_reactant, reax)
                if tag == 1 and pairs_reactant.shape[0] > 0:
                    self.add_to_atom_for_reaction(-1, 1, pairs_reactant, reax, pairs_reactant)
        else:
            for atom in atoms_as_reactant_1_for_reax:
                idx_in_bonded_atoms = np.where(bonded_atoms[0] == atom)[0]
                atoms_bonded = bonded_atoms[1][idx_in_bonded_atoms]
                reactant_atoms_bonded = bonded_atoms[1][
                    idx_in_bonded_atoms[np.where((self.all_atoms_features[atoms_bonded] == reactant_2).all(-1))[0]]]
                modif_h[reax] += reactant_atoms_bonded.shape[0]
                if self.MD_or_KMC == 1:
                    pairs_reactant_atom = np.vstack(
                        (np.ones(reactant_atoms_bonded.shape[0], dtype=int) * atoms_involved[atom],
                         reactant_atoms_bonded)).T
                    pairs_reactant = np.vstack((pairs_reactant, pairs_reactant_atom))
            for atom in atoms_as_reactant_2_for_reax:
                idx_in_bonded_atoms = np.where(bonded_atoms[0] == atom)[0]
                atoms_bonded = bonded_atoms[1][idx_in_bonded_atoms]
                reactant_atoms_bonded = bonded_atoms[1][
                    idx_in_bonded_atoms[np.where((self.all_atoms_features[atoms_bonded] == reactant_1).all(-1))[0]]]
                modif_h[reax] += reactant_atoms_bonded.shape[0]
                if self.MD_or_KMC == 1:
                    pairs_reactant_atom = np.vstack((reactant_atoms_bonded, (
                            np.ones(reactant_atoms_bonded.shape[0], dtype=int) * atoms_involved[atom]))).T
                    pairs_reactant = np.vstack((pairs_reactant, pairs_reactant_atom))
            modif_h[reax] -= np.count_nonzero(bonded_atoms_involved_reax)
            if self.MD_or_KMC == 1:
                if pairs_reactant.shape[0] > 0:
                    if tag == 0:
                        self.remove_from_atom_for_reaction(0, 1, pairs_reactant, reax)
                    elif tag == 1:
                        self.add_to_atom_for_reaction(0, 1, pairs_reactant, reax, pairs_reactant)
        modif_num_of_reactants[reax, 0] = modif_h[reax]

    def modif_h_creation_different_atoms(self, atoms_as_reactant_1_for_reax, atoms_as_reactant_2_for_reax, bonded_atoms,
                                         atoms_involved, modif_num_of_reactants, modif_h, reax,
                                         bonded_atoms_involved, tag):
        # Compute the modifications in h, num_of_reactants, and atom_for_reaction before (tag2 == 0) or after (tag2
        # == 1) for a reaction that create a bond between 2 atoms with different descriptions.
        reactant_1 = \
            self.reaction_dict[reax,
            self.num_reaction_descriptors:(self.num_atom_descriptors + self.num_reaction_descriptors)]
        reactant_2 = \
            self.reaction_dict[reax,
            (self.num_atom_descriptors + self.num_reaction_descriptors):
            2 * (self.num_atom_descriptors + self.num_reaction_descriptors)]
        bonded_atoms_involved_reax = bonded_atoms_involved[atoms_as_reactant_1_for_reax, :]
        bonded_atoms_involved_reax = bonded_atoms_involved_reax[:, atoms_as_reactant_2_for_reax]
        pairs_reactant_1 = np.array([], dtype=int)
        pairs_reactant_1.shape = [0, 2]
        pairs_reactant_2 = np.array([], dtype=int)
        pairs_reactant_2.shape = [0, 2]
        modif_h[reax], pairs_reactant_1 = \
            self.modif_h_due_to_reactant(atoms_as_reactant_1_for_reax, modif_h[reax], reax, bonded_atoms, reactant_2, 1,
                                         atoms_involved, pairs_reactant_1, tag)
        modif_h[reax], pairs_reactant_2 = \
            self.modif_h_due_to_reactant(atoms_as_reactant_2_for_reax, modif_h[reax], reax, bonded_atoms, reactant_1, 0,
                                         atoms_involved, pairs_reactant_2, tag)
        modif_num_of_reactants[reax, 0] = atoms_as_reactant_1_for_reax.shape[0]
        modif_num_of_reactants[reax, 1] = atoms_as_reactant_2_for_reax.shape[0]
        if tag == 0:
            # If atoms_involved participating in the reaction are not bonded, they are counted twice
            modif_h[reax] -= np.count_nonzero(bonded_atoms_involved_reax == 0)
        elif tag == 1:
            modif_h[reax] += np.count_nonzero(bonded_atoms_involved_reax == 0) + 2 * np.count_nonzero(
                bonded_atoms_involved_reax)
        if self.MD_or_KMC == 1:
            if modif_h[reax] > 0:
                if tag == 0:
                    if atoms_as_reactant_1_for_reax.shape[0] > 0:
                        self.remove_from_atom_for_reaction(0, 0, atoms_involved[atoms_as_reactant_1_for_reax], reax)
                    if atoms_as_reactant_2_for_reax.shape[0] > 0:
                        self.remove_from_atom_for_reaction(1, 0, atoms_involved[atoms_as_reactant_2_for_reax], reax)
                elif tag == 1:
                    if atoms_as_reactant_1_for_reax.shape[0] > 0:
                        self.add_to_atom_for_reaction(0, 0, atoms_involved[atoms_as_reactant_1_for_reax], reax,
                                                      pairs_reactant_1)
                    if atoms_as_reactant_2_for_reax.shape[0] > 0:
                        self.add_to_atom_for_reaction(1, 0, atoms_involved[atoms_as_reactant_2_for_reax], reax,
                                                      pairs_reactant_2,
                                                      atoms_to_be_careful=atoms_involved[atoms_as_reactant_1_for_reax])

    def modif_h_due_to_reactant(self, atoms_as_reactant_for_reax, old_modif_h_reax, reax, bonded_atoms, other_reactant,
                                other_reactant_number, atoms_involved, old_pairs_reactant, tag):
        modif_h_reax = old_modif_h_reax.copy()
        pairs_reactant = old_pairs_reactant.copy()
        for atom in atoms_as_reactant_for_reax:
            if self.reaction_type[reax] == 2:
                if tag == 0:
                    modif_h_reax += self.num_of_reactants[reax, other_reactant_number] - 1
                elif tag == 1:
                    modif_h_reax += self.num_of_reactants[reax, other_reactant_number]
            else:
                modif_h_reax += self.num_of_reactants[reax, other_reactant_number]
            idx_in_bonded_atoms = np.where(bonded_atoms[0] == atom)[0]
            atoms_bonded = bonded_atoms[1][idx_in_bonded_atoms]
            reactant_atoms_bonded = bonded_atoms[1][
                idx_in_bonded_atoms[np.where((self.all_atoms_features[atoms_bonded] == other_reactant).all(-1))[0]]]
            modif_h_reax -= reactant_atoms_bonded.shape[0]
            if self.MD_or_KMC == 1:
                if other_reactant_number == 1:
                    pairs_reactant_atom = np.vstack((
                                                    np.ones(reactant_atoms_bonded.shape[0], dtype=int) * atoms_involved[
                                                        atom], reactant_atoms_bonded)).T
                else:
                    pairs_reactant_atom = np.vstack((reactant_atoms_bonded, (
                                np.ones(reactant_atoms_bonded.shape[0], dtype=int) * atoms_involved[atom]))).T
                pairs_reactant = np.vstack((pairs_reactant, pairs_reactant_atom))
        return [modif_h_reax, pairs_reactant]

    def modif_h_creation_same_atoms(self, atoms_as_reactant_1_for_reax, bonded_atoms, atoms_involved,
                                    modif_num_of_reactants, modif_h, reax, bonded_atoms_involved, tag):
        # Compute the modifications in h, num_of_reactants, and atom_for_reaction before (tag2 == 0) or after (tag2
        # == 1) for a reaction that create a bond between 2 atoms with the same descriptions.
        reactant_1 = self.reaction_dict[reax,
                     self.num_reaction_descriptors:(self.num_atom_descriptors + self.num_reaction_descriptors)]
        bonded_atoms_involved_reax = bonded_atoms_involved[atoms_as_reactant_1_for_reax, :]
        bonded_atoms_involved_reax = bonded_atoms_involved_reax[:, atoms_as_reactant_1_for_reax]
        pairs_reactant = np.array([], dtype=int)
        pairs_reactant.shape = [0, 2]
        modif_h[reax], pairs_reactant = self.modif_h_due_to_reactant(atoms_as_reactant_1_for_reax, modif_h[reax], reax,
                                                                     bonded_atoms, reactant_1, 0, atoms_involved,
                                                                     pairs_reactant, tag)
        modif_num_of_reactants[reax, 0] = atoms_as_reactant_1_for_reax.shape[0]
        if tag == 0:
            modif_h[reax] -= (np.count_nonzero(bonded_atoms_involved_reax == 0) - atoms_as_reactant_1_for_reax.shape[
                0]) // 2
        elif tag == 1:
            modif_h[reax] += (np.count_nonzero(bonded_atoms_involved_reax == 0) - atoms_as_reactant_1_for_reax.shape[
                0]) // 2 + np.count_nonzero(bonded_atoms_involved_reax)
        if self.MD_or_KMC == 1:
            if modif_h[reax] > 0:
                if tag == 0:
                    self.remove_from_atom_for_reaction(-1, 0, atoms_involved[atoms_as_reactant_1_for_reax], reax)
                elif tag == 1:
                    self.add_to_atom_for_reaction(-1, 0, atoms_involved[atoms_as_reactant_1_for_reax], reax,
                                                  pairs_reactant)

    def remove_from_atom_for_reaction(self, position, pairs, atoms_to_remove, reax):
        # Remove from atom_for_reaction the pairs that can't happen anymore.
        if pairs == 1:
            if position == -1:
                atoms_to_remove = np.sort(atoms_to_remove)
            if atoms_to_remove.shape[0] > 0:
                atoms_to_remove = np.unique(atoms_to_remove, axis=0)
                self.atom_for_reactions[reax] = np.delete(self.atom_for_reactions[reax], np.where(
                    (self.atom_for_reactions[reax] == atoms_to_remove[:, None]).all(-1))[1], axis=0)
        elif pairs == 0:
            if position == -1:
                self.atom_for_reactions[reax] = np.delete(self.atom_for_reactions[reax], np.where(
                    (self.atom_for_reactions[reax][:, None] == atoms_to_remove[:, None]).any(1).any(1))[0], axis=0)
            else:
                self.atom_for_reactions[reax] = np.delete(self.atom_for_reactions[reax], np.where(
                    self.atom_for_reactions[reax][:, position] == atoms_to_remove[:, None])[1], axis=0)

    def add_to_atom_for_reaction(self, position, pairs, atoms_to_add, reax, pairs_bonded,
                                 atoms_to_be_careful=np.array([])):
        # Add to atom_for_reaction the pairs that can happen for new reactions.
        if pairs == 1:
            if position == -1:
                atoms_to_add = np.sort(atoms_to_add)
            if atoms_to_add.shape[0] > 0:
                atoms_to_add = np.unique(atoms_to_add, axis=0)
                self.atom_for_reactions[reax] = np.vstack((self.atom_for_reactions[reax], atoms_to_add))
        elif pairs == 0:
            if position == 0:
                other_reactant = self.reaction_dict[reax][
                                 (self.num_atom_descriptors + self.num_reaction_descriptors):2 * (
                                             self.num_atom_descriptors + self.num_reaction_descriptors)]
                other_reactant_atoms = np.where((self.all_atoms_features == other_reactant).all(-1))[0]
                pairs_to_add = np.vstack((np.repeat(atoms_to_add, other_reactant_atoms.shape[0]),
                                          np.tile(other_reactant_atoms, atoms_to_add.shape[0]))).T
                pairs_to_add = np.delete(pairs_to_add, np.where((pairs_to_add == pairs_bonded[:, None]).all(-1))[1],
                                         axis=0)
                self.atom_for_reactions[reax] = np.vstack((self.atom_for_reactions[reax], pairs_to_add))
            elif position == 1:
                other_reactant = self.reaction_dict[reax][self.num_reaction_descriptors:(
                            self.num_atom_descriptors + self.num_reaction_descriptors)]
                other_reactant_atoms = np.where((self.all_atoms_features == other_reactant).all(-1))[0]
                if atoms_to_be_careful.shape[0] > 0:
                    atoms_to_be_careful_idx = np.argwhere(other_reactant_atoms[:, None] == atoms_to_be_careful)[:, 0]
                    other_reactant_atoms = np.delete(other_reactant_atoms, atoms_to_be_careful_idx)
                pairs_to_add = np.vstack((np.tile(other_reactant_atoms, atoms_to_add.shape[0]),
                                          np.repeat(atoms_to_add, other_reactant_atoms.shape[0]))).T
                pairs_to_add = np.delete(pairs_to_add, np.where((pairs_to_add == pairs_bonded[:, None]).all(-1))[1],
                                         axis=0)
                self.atom_for_reactions[reax] = np.vstack((self.atom_for_reactions[reax], pairs_to_add))
            elif position == -1:

                other_reactant = self.reaction_dict[reax][self.num_reaction_descriptors:(
                            self.num_atom_descriptors + self.num_reaction_descriptors)]
                other_reactant_atoms = np.where((self.all_atoms_features == other_reactant).all(-1))[0]
                pairs_to_add = np.vstack((np.tile(other_reactant_atoms, atoms_to_add.shape[0]),
                                          np.repeat(atoms_to_add, other_reactant_atoms.shape[0]))).T
                pairs_to_add = np.delete(pairs_to_add, np.where(pairs_to_add[:, 0] == pairs_to_add[:, 1]), axis=0)
                pairs_to_add = np.sort(pairs_to_add, axis=1)
                pairs_to_add = np.unique(pairs_to_add, axis=0)
                if pairs_bonded.shape[0] > 0:
                    pairs_bonded = np.sort(pairs_bonded, axis=1)
                    pairs_bonded = np.unique(pairs_bonded, axis=0)
                    pairs_to_add = np.delete(pairs_to_add, np.where((pairs_to_add == pairs_bonded[:, None]).all(-1))[1],
                                             axis=0)
                self.atom_for_reactions[reax] = np.vstack((self.atom_for_reactions[reax], pairs_to_add))

    def update_bond_matrix(self, bond_matrix, bond_change):
        # Update the adjacency matrix after some bond changes.
        for reax in range(bond_change.shape[0]):
            bond_matrix[bond_change[reax][0], bond_change[reax][1]] = bond_change[reax][2]
            bond_matrix[bond_change[reax][1], bond_change[reax][0]] = bond_change[reax][2]

    def make_reaction(self, reaction_to_happen, h, t, bond_change, bond_matrix):
        # Update propensity, bond_change_KMC and the bond_matrix
        atoms_involved = self.pick_atoms_involved(reaction_to_happen)
        atoms_involved = np.sort(atoms_involved)
        bond_change[t] = np.append(atoms_involved, [self.reaction_dict[reaction_to_happen, 0],
                                                    reaction_to_happen]).reshape((-1, 1)).T
        h = self.update_all_after_reax(h, bond_matrix, bond_change[t], t)
        return [h, bond_change, bond_matrix]

    def pick_atoms_involved(self, reaction_to_happen):
        # Randomly pick the atoms involved in the reaction
        idx_atoms_involved = np.random.randint(self.atom_for_reactions[reaction_to_happen].shape[0])
        atoms_involved = self.atom_for_reactions[reaction_to_happen][idx_atoms_involved, :]
        return atoms_involved

