import numpy as np
import time
import scipy


class Reactions:
    def __init__(self, reaction_dict, molecules, molecule_list):
        self.reaction_occurrences = np.array([], dtype=int)  # Number of times a reaction happen
        if reaction_dict.shape[0] != 0:
            self.reaction_dict = reaction_dict
            self.reactant_dict = - self.reaction_dict.copy()
            self.reactant_dict[self.reactant_dict < 0] = 0
        else:
            self.reaction_dict = np.zeros([50000, 50000], dtype=int)
            self.reactant_dict = np.array([], dtype=int)
        if self.reactant_dict.shape[0] != 0:
            self.reactant_reax = {}
            self.reactant_repeat = np.zeros([self.reactant_dict.shape[0]], dtype=int)
            for reax in range(self.reactant_dict.shape[0]):
                self.reactant_reax[reax] = np.where(self.reactant_dict[reax, :])
                if (self.reactant_dict[reax, self.reactant_reax[reax]] > 1).any():
                    self.reactant_repeat[reax] = 1
        self.molecules = molecules
        self.molecule_list = molecule_list
        self.molecule_per_frame = np.zeros([50000, molecule_list.shape[0]], dtype=int)
        self.time_range = np.array([])
        self.number_of_reaction = 0
        self.occurrences_per_frame = {}

    def update_reactions_and_real_bonds(self, reaction_frame, real_bonds, frame, molecules, old_molecules_frame,
                                        old_molecule_bond_change, new_molecules_frame, new_molecule_bond_change):
        # Here this function only update real_bonds because the reactions will be updated later
        [reactants_molecules, products_molecules] = self.get_independent_reactions(old_molecule_bond_change,
                                                                                   new_molecule_bond_change)
        for reax in range(len(reactants_molecules)):
            reaction_description = self.get_reaction_description(reactants_molecules, products_molecules,
                                                                 old_molecules_frame, new_molecules_frame, molecules,
                                                                 reax)
            reaction_index = np.where((self.reaction_dict[:self.number_of_reaction,
                                       :reaction_description.shape[0]] == reaction_description).all(-1))[0]
            if reaction_index.shape[0] != 0:
                self.reaction_occurrences[reaction_index[0]] += 1
                self.occurrences_per_frame[reaction_index[0]].append(frame)
            else:
                self.reaction_dict[self.number_of_reaction, :reaction_description.shape[0]] = reaction_description
                self.reaction_occurrences = np.append(self.reaction_occurrences, 1)
                self.occurrences_per_frame[self.number_of_reaction] = [frame]
                self.number_of_reaction += 1
        for reax in range(reaction_frame.shape[0]):
            real_bonds[reaction_frame[reax, 0], reaction_frame[reax, 1]] = reaction_frame[reax, 2]

    def get_independent_reactions(self, old_molecule_bond_change, new_molecule_bond_change):
        # Obtain independent reactions, i.e. the reactions that involve the same set of atoms.
        reactants_molecules = []
        products_molecules = []
        reactants_involved = old_molecule_bond_change[:, :2]
        products_involved = new_molecule_bond_change[:, :2]
        to_delete = []
        for row in range(reactants_involved.shape[0]):
            found = False
            for i in range(len(reactants_molecules)):
                if any(x in reactants_molecules[i] for x in reactants_involved[row, :]):
                    if found == False:
                        reactants_molecules[i].update(reactants_involved[row, :])
                        products_molecules[i].update(products_involved[row, :])
                        found = True
                        already_found_idx = i
                    else:
                        reactants_molecules[already_found_idx].update(reactants_molecules[i])
                        products_molecules[already_found_idx].update(products_molecules[i])
                        to_delete.append(i)
            if found == False:
                reactants_molecules.append(set(reactants_involved[row, :2]))
                products_molecules.append(set(products_involved[row, :2]))
        if len(to_delete) != 0:
            to_delete = np.unique(to_delete)[::-1]
            for i in to_delete:
                del reactants_molecules[i]
                del products_molecules[i]
        return [reactants_molecules, products_molecules]

    def get_reaction_description(self, reactants_molecules, products_molecules, old_molecules_frame,
                                 new_molecules_frame, molecules, reax):
        # Obtain the reaction description with the molecular features
        reaction_description = np.zeros([molecules.molecule_list.shape[0]])
        for mol in reactants_molecules[reax]:
            reaction_description[np.where((molecules.molecule_list == old_molecules_frame[mol]).all(-1))] += -1
        for mol in products_molecules[reax]:
            reaction_description[np.where((molecules.molecule_list == new_molecules_frame[mol]).all(-1))] += 1
        return reaction_description

    def truncate_reaction_dict(self):
        # Some arrays are initialized with too many rows or columns. This function truncates the arrays to the size
        # they should be.
        self.reaction_dict = self.reaction_dict[:self.number_of_reaction, :self.molecules.molecule_list.shape[0]]
        self.reaction_occurrences = np.delete(self.reaction_occurrences, np.where((self.reaction_dict == 0).all(-1)),
                                              axis=0)
        self.reaction_dict = np.delete(self.reaction_dict, np.where((self.reaction_dict == 0).all(-1)), axis=0)
        self.reactant_dict = - self.reaction_dict.copy()
        self.reactant_dict[self.reactant_dict < 0] = 0

    def get_h(self, molecule_per_frame, timerange, start_frame_analysis, end_frame_analysis):
        # Get the number of times each reaction could have happened (h) starting from the state of the system defined
        # by "first_frame" at the frame "start_frame" until the "end_frame".
        start_index_analysis = np.where(timerange >= start_frame_analysis)[0][0]
        end_index_analysis = np.where(timerange <= end_frame_analysis)[0][-1]
        h_per_frame = np.zeros([end_index_analysis - start_index_analysis + 1, self.reactant_dict.shape[0]])
        comb_func = np.vectorize(scipy.special.comb)
        print("Total number of reactions: {}".format(self.reactant_dict.shape[0]), flush=True)
        percent = 0.1
        coeff_per_frame = timerange[start_index_analysis + 1:end_index_analysis + 1] - timerange[
                                                                                       start_index_analysis:end_index_analysis]
        coeff_per_frame = np.append(coeff_per_frame, end_frame_analysis - timerange[end_index_analysis])
        for reax in range(self.reactant_dict.shape[0]):
            if reax == int(percent * self.reactant_dict.shape[0]):
                print(reax, flush=True)
                print(time.process_time(), flush=True)
                percent += 0.1
            reactant_reax = np.where(self.reactant_dict[reax, :])
            if (self.reactant_dict[reax, reactant_reax] > 1).any():
                h_per_frame[:, reax] = coeff_per_frame * np.prod(
                    comb_func(molecule_per_frame[start_index_analysis:end_index_analysis + 1, reactant_reax],
                              self.reactant_dict[reax, reactant_reax]), axis=(1, 2))
            else:
                h_per_frame[:, reax] = coeff_per_frame * np.prod(
                    molecule_per_frame[start_index_analysis:end_index_analysis + 1, reactant_reax], axis=(1, 2))
        h_tot = np.sum(h_per_frame, axis=0)
        return [h_tot, h_per_frame]

    def get_h_frame(self, bond_matrix):
        # We get the number of times h each reaction could have happened for one frame.
        [molecule_frame, molecule_list_frame, self.time_range] = self.molecules.get_molecule_full(bond_matrix, {}, 0)
        idx_transfer = np.where((self.molecule_list[:, None] == molecule_list_frame).all(-1))
        self.molecule_per_frame[0, idx_transfer[0]] = molecule_frame[:, idx_transfer[1]]
        comb_func = np.vectorize(scipy.special.comb)
        h = np.zeros([self.reactant_dict.shape[0]])
        for reax in range(self.reactant_dict.shape[0]):
            if self.reactant_repeat[reax] == 1:
                h[reax] = np.prod(comb_func(self.molecule_per_frame[0, self.reactant_reax[reax]],
                                            self.reactant_dict[reax, self.reactant_reax[reax]]), axis=1)
            else:
                h[reax] = np.prod(self.molecule_per_frame[0, self.reactant_reax[reax]], axis=1)
        return h

    def make_reaction(self, reaction_to_happen, h, t, bond_change, bond_matrix):
        # Update propensity and the bond_matrix
        self.time_range = np.append(self.time_range, t)
        if self.time_range.shape[0] > self.molecule_per_frame.shape[0]:
            molecule_per_frame_temp = self.molecule_per_frame.copy()
            self.molecule_per_frame = np.zeros([self.molecule_per_frame.shape[0] * 2, self.molecule_per_frame.shape[1]])
            self.molecule_per_frame[:molecule_per_frame_temp.shape[0], :] = molecule_per_frame_temp
        self.molecule_per_frame[self.time_range.shape[0] - 1, :] = self.molecule_per_frame[self.time_range.shape[0] - 2,
                                                                   :] + self.reaction_dict[reaction_to_happen, :]
        comb_func = np.vectorize(scipy.special.comb)
        molecules_involved = np.where(self.reaction_dict[reaction_to_happen])
        reactions_modified = np.where((self.reactant_dict[:, molecules_involved]).any(-1))[0]
        for reax in reactions_modified:
            if self.reactant_repeat[reax] == 1:
                h[reax] = np.prod(
                    comb_func(self.molecule_per_frame[self.time_range.shape[0] - 1, self.reactant_reax[reax]],
                              self.reactant_dict[reax, self.reactant_reax[reax]]),
                    axis=1)
            else:
                h[reax] = np.prod(self.molecule_per_frame[self.time_range.shape[0] - 1, self.reactant_reax[reax]],
                                  axis=1)
        return [h, {}, bond_matrix]
