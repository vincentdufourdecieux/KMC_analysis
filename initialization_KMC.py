import numpy as np

import atoms
import molecule_creation
import reactions_atomic
import reactions_molecular
import save_data


def get_all_for_KMC(foldername_reactions, foldername_starting_molecules):
    first_frame, num_of_atoms, start_frame_MD, end_frame_MD, atom_types, timestep = save_data.load_MD_parameters_starting_molecules(
        foldername_starting_molecules)
    num_of_type, reaction_framework, molecule_list = save_data.load_MD_parameters_reactions(
        foldername_reactions)

    atoms_KMC = atoms.Atoms(num_of_type, atom_types)
    molecules_KMC = molecule_creation.MoleculeList(num_of_atoms, np.array([]), atom_types, atoms_KMC, num_of_type,
                                                   start_frame_MD,
                                                   end_frame_MD)
    if reaction_framework == 'atomic':
        df_reactions = save_data.load_MD_data_atomic(foldername_reactions)
        reactions_KMC = reactions_atomic.Reactions(num_of_type,
                                                   np.array(df_reactions.iloc[:, 5:]), np.array(df_reactions["Type"]),
                                                   atoms_KMC, 1, molecules_KMC)
    elif reaction_framework == 'molecular':
        df_reactions = save_data.load_MD_data_molecular(foldername_reactions)
        reaction_dict = get_reaction_dict_molecules_framework(df_reactions[["Reactants", "Products"]], molecule_list)
        reactions_KMC = reactions_molecular.Reactions(reaction_dict, molecules_KMC, molecule_list)
    end_t = (end_frame_MD - start_frame_MD) * timestep
    reaction_rates = np.array(df_reactions["Rate"])

    return [reactions_KMC, atoms_KMC, molecules_KMC, end_t, reaction_rates, reaction_framework, first_frame]


def get_reaction_dict_molecules_framework(df_reactions, molecule_list):
    reaction_dict = np.zeros([df_reactions.shape[0], molecule_list.shape[0]])
    for i in range(reaction_dict.shape[0]):
        for j in range(len(df_reactions["Reactants"].iloc[i])):
            reaction_dict[i, df_reactions["Reactants"].iloc[i][j][0]] = - df_reactions["Reactants"].iloc[i][j][1]
        for j in range(len(df_reactions["Products"].iloc[i])):
            reaction_dict[i, df_reactions["Products"].iloc[i][j][0]] = df_reactions["Products"].iloc[i][j][1]
    return reaction_dict

