import numpy as np
from scipy import sparse
import time


class BondAnalysis:
    def __init__(self, pipeline, bond_duration, num_of_atoms):
        self.pipeline = pipeline
        self.bond_duration = bond_duration
        self.num_of_atoms = num_of_atoms

    def run_analysis(self, reactions, molecules, start_frame_MD, end_frame_MD, start_frame_analysis,
                     end_frame_analysis):
        # Run the molecular analyzer, for each frame, check if a reaction happens using the bond criterion. If yes,
        # store the reactions, the number of times it happened, its type... Keep the first frame and the bond changes.
        real_bonds = sparse.dok_matrix((self.num_of_atoms, self.num_of_atoms))
        bond_follow = sparse.csr_matrix((self.num_of_atoms, self.num_of_atoms))
        bond_change = {}
        for frame in range(start_frame_MD, end_frame_MD):
            if frame % 100 == 0:
                print(frame, flush=True)
                print(time.process_time(), flush=True)
            data = self.pipeline.compute(frame)
            [real_bonds, bond_follow, reaction_frame] = self.apply_bond_duration_criterion(
                real_bonds, bond_follow, frame, data.particles.bonds.topology, data.particles.identifiers,
                start_frame_MD)
            if frame == start_frame_MD:
                first_frame = real_bonds.copy()
                molecules.initialize_molecules((first_frame.T + first_frame).toarray(), frame)
            if frame == start_frame_analysis:
                first_frame_analysis = real_bonds.copy()
            if reaction_frame.shape[0] != 0:
                reaction_frame = reaction_frame[reaction_frame[:, 0].argsort()]
                bond_change[frame + 1 - self.bond_duration] = reaction_frame
                [old_molecules_frame, old_molecule_bond_change, new_molecules_frame,
                 new_molecule_bond_change] = molecules.update_after_reaction(frame + 1 - self.bond_duration,
                                                                             bond_change[
                                                                                 frame + 1 - self.bond_duration])
                if start_frame_analysis <= frame <= end_frame_analysis:
                    reactions.update_reactions_and_real_bonds(reaction_frame, real_bonds,
                                                              frame + 1 - self.bond_duration, molecules,
                                                              old_molecules_frame, old_molecule_bond_change,
                                                              new_molecules_frame, new_molecule_bond_change)
                else:
                    for reax in range(reaction_frame.shape[0]):
                        real_bonds[reaction_frame[reax, 0], reaction_frame[reax, 1]] = reaction_frame[reax, 2]
        return [first_frame, bond_change, first_frame_analysis]

    def apply_bond_duration_criterion(self, real_bonds, bond_follow, frame, data_bonds, data_identifiers,
                                      start_frame_MD):
        # Apply the bond duration criteria. Check if a bond has been broken or created for "bond_duration" number of
        # steps, otherwise put back the count to 0. If a reaction happens, store it in reaction_frame.
        reaction_frame = np.array([], dtype=int)
        reaction_frame.shape = [0, 4]
        data_bonds_sparse = sparse.dok_matrix((self.num_of_atoms, self.num_of_atoms))
        kk = data_identifiers[data_bonds] - 1
        kk.sort()
        data_bonds_sparse[kk[:, 0], kk[:, 1]] = 1
        if frame == start_frame_MD:
            real_bonds = data_bonds_sparse
        else:
            abs_difference = abs(real_bonds - data_bonds_sparse)
            bond_follow = bond_follow.multiply(abs_difference) + abs_difference
        reaction_happening = sparse.find(bond_follow == self.bond_duration)
        if reaction_happening[0].shape[0] != 0:
            for reax in range(reaction_happening[0].shape[0]):
                atom_1 = reaction_happening[0][reax]
                atom_2 = reaction_happening[1][reax]
                reaction_frame = np.append(reaction_frame, [[atom_1, atom_2, int(1 - real_bonds[atom_1, atom_2]), 0]],
                                           0)
                bond_follow[atom_1, atom_2] = 0
        return [real_bonds, bond_follow, reaction_frame]
