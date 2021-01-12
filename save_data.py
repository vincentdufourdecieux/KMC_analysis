import numpy as np
import pandas as pd
import pickle
from scipy import sparse
import os
import pdb

def save_MD_data_atomic(foldername, reaction_rates, reaction_type, reaction_occurrences, h, reaction_dict,
                   start_frame_MD, end_frame_MD, num_of_type, num_of_atoms, atom_types, timestep,
                   reaction_framework, bond_change, first_frame, molecule_per_frame, molecule_list, timerange):
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    df_rate = pd.DataFrame({"Rate": reaction_rates}, dtype=float)
    df_type = pd.DataFrame({"Type": reaction_type}, dtype=int)
    df_occurrences = pd.DataFrame({"Occurrences": reaction_occurrences}, dtype=int)
    df_h = pd.DataFrame({"h": h}, dtype=int)

    df_features = pd.DataFrame(reaction_dict, dtype=int)
    df_reax = pd.concat([df_rate, df_type, df_occurrences, df_h, df_features], axis=1)
    df_reax.to_csv("".join([foldername, "Reaction.csv"]))

    dict_parameters = {'Start frame MD': start_frame_MD, 'End frame MD': end_frame_MD, 'Number of Types': num_of_type,
                       'Number of Atoms': num_of_atoms, 'Atom_types': atom_types, 'Timestep': timestep,
                       'Reaction framework': reaction_framework}
    save_dict(foldername, "Parameters.pkl", dict_parameters)
    save_dict(foldername, "Bond_change.pkl", bond_change)

    sparse.save_npz("".join([foldername, "First_frame.npz"]), sparse.coo_matrix(first_frame))

    sparse.save_npz("".join([foldername, "Molecules_per_frame.npz"]), sparse.coo_matrix(molecule_per_frame))
    np.savetxt("".join([foldername, "Molecule_list"]), molecule_list)
    np.savetxt("".join([foldername, "Time_range"]), timerange)

def save_MD_data_molecular(foldername, reaction_rates, reaction_occurrences, h, reaction_dict, molecule_list,
                   start_frame_MD, end_frame_MD, num_of_type, num_of_atoms, atom_types, timestep,
                   reaction_framework, bond_change, first_frame, molecule_per_frame, timerange):
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    df_rate = pd.DataFrame({"Rate": reaction_rates}, dtype=float)
    df_occurrences = pd.DataFrame({"Occurrences": reaction_occurrences}, dtype=int)
    df_h = pd.DataFrame({"h": h}, dtype=int)
    df_features = get_features_for_molecules_framework(reaction_dict)
    react_strings = get_reactions_strings(molecule_list, reaction_dict)
    df_react_strings = pd.DataFrame({"Reaction strings": react_strings}, dtype=object)
    df_reax = pd.concat([df_rate, df_occurrences, df_h, df_react_strings, df_features], axis=1)
    df_reax.to_csv("".join([foldername, "Reaction.csv"]))

    dict_parameters = {'Start frame MD': start_frame_MD, 'End frame MD': end_frame_MD, 'Number of Types': num_of_type,
                       'Number of Atoms': num_of_atoms, 'Atom_types': atom_types, 'Timestep': timestep,
                       'Reaction framework': reaction_framework}
    save_dict(foldername, "Parameters.pkl", dict_parameters)
    save_dict(foldername, "Bond_change.pkl", bond_change)

    sparse.save_npz("".join([foldername, "First_frame.npz"]), sparse.coo_matrix(first_frame))

    sparse.save_npz("".join([foldername, "Molecules_per_frame.npz"]), sparse.coo_matrix(molecule_per_frame))
    np.savetxt("".join([foldername, "Molecule_list"]), molecule_list)
    np.savetxt("".join([foldername, "Time_range"]), timerange)


def save_dict(foldername, fname, dict_to_save):
    output = open("".join([foldername, fname]), 'wb')
    pickle.dump(dict_to_save, output)
    output.close()

def load_parameters(foldername):
    pkl_file = open("".join([foldername, "Parameters.pkl"]), 'rb')
    parameters = pickle.load(pkl_file)
    pkl_file.close()

    atom_features = parameters['Atom Features']
    reaction_features = parameters['Reaction Features']
    start_frame_MD = parameters['Start frame MD']
    end_frame_MD = parameters['End frame MD']
    num_of_type = parameters['Number of Types']
    num_of_atoms = parameters['Number of Atoms']
    atom_types = parameters['Atom_types']
    timestep = parameters['Timestep']
    reaction_framework = parameters['Reaction framework']
    molecule_list = np.loadtxt("".join([foldername, "Molecule_list"]))

    first_frame = sparse.load_npz("".join([foldername, "First_frame.npz"]))
    first_frame = first_frame.todok()
    first_frame = first_frame.toarray()

    return [atom_features, reaction_features, start_frame_MD, end_frame_MD, num_of_type, num_of_atoms, atom_types,
            timestep, reaction_framework, molecule_list, first_frame]

def load_MD_parameters_starting_molecules(foldername):
    first_frame = sparse.load_npz("".join([foldername, "First_frame.npz"]))
    first_frame = first_frame.todok()
    first_frame = first_frame.toarray()

    pkl_file = open("".join([foldername, "Parameters.pkl"]), 'rb')
    parameters = pickle.load(pkl_file)
    pkl_file.close()

    start_frame_MD = parameters['Start frame MD']
    end_frame_MD = parameters['End frame MD']
    num_of_atoms = parameters['Number of Atoms']
    atom_types = parameters['Atom_types']
    timestep = parameters['Timestep']
    return first_frame, num_of_atoms, start_frame_MD, end_frame_MD, atom_types, timestep

def load_MD_parameters_reactions(foldername):
    pkl_file = open(foldername + "Parameters.pkl", 'rb')
    parameters = pickle.load(pkl_file)
    pkl_file.close()

    num_of_type = parameters['Number of Types']
    reaction_framework = parameters['Reaction framework']
    molecule_list = np.loadtxt("".join([foldername, "Molecule_list"]))
    return num_of_type, reaction_framework, molecule_list

def load_MD_data_atomic(foldername):
    df_reactions = pd.read_csv(foldername +  "Reaction.csv")
    return df_reactions

def load_MD_data_molecular(foldername):
    df_reactions = pd.read_csv("".join([foldername, "Reaction.csv"]), converters={"Reactants": eval, "Products": eval})
    return df_reactions

def load_dict(foldername, fname):
    pkl_file = open("".join([foldername, fname]), 'rb')
    dict_to_load = pickle.load(pkl_file)
    pkl_file.close()
    return dict_to_load

def save_data_KMC(foldername, bond_change_KMC, molecule_per_frame, molecule_list, timerange_KMC, num):
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    output = open("".join([foldername, "Bond_change_KMC_", str(num), ".pkl"]), 'wb')
    pickle.dump(bond_change_KMC, output)
    output.close()
    sparse.save_npz("".join([foldername, "Molecules_per_frame_KMC_", str(num), ".npz"]),
                    sparse.coo_matrix(molecule_per_frame))
    np.savetxt("".join([foldername, "Molecule_list_KMC_", str(num)]), molecule_list)
    np.savetxt("".join([foldername, "Time_range_KMC_", str(num)]), timerange_KMC)

def get_reactions_strings(molecule_list, reaction_dict):
    react_strings = np.zeros([reaction_dict.shape[0]], dtype='object')
    for i in range(reaction_dict.shape[0]):
        reactants = np.where(reaction_dict[i] < 0)
        products = np.where(reaction_dict[i] > 0)
        molecules_reactants = np.hstack((molecule_list[reactants], abs(reaction_dict[i, reactants]).T))
        molecules_products = np.hstack((molecule_list[products], abs(reaction_dict[i, products]).T))
        molecules_reactants = molecules_reactants[np.lexsort(molecules_reactants[:, range(4, -1, -1)].T)][::-1]
        molecules_products = molecules_products[np.lexsort(molecules_products[:, range(4, -1, -1)].T)][::-1]
        react_string = ''
        for j in range(molecules_reactants.shape[0]):
            for k in range(int(molecules_reactants[j, -1])):
                if molecule_list.shape[1] == 5:
                    if molecules_reactants[j, 0] != 0:
                        react_string += 'C' + str(int(molecules_reactants[j, 0])) + ' '
                    if molecules_reactants[j, 1] != 0:
                        react_string += 'H' + str(int(molecules_reactants[j, 1])) + ' '
                    if molecules_reactants[j, 2] != 0:
                        react_string += str(int(molecules_reactants[j, 2])) + '(C-C)' + ' '
                    if molecules_reactants[j, 3] != 0:
                        react_string += str(int(molecules_reactants[j, 3])) + '(C-H)' + ' '
                    if molecules_reactants[j, 4] != 0:
                        react_string += str(int(molecules_reactants[j, 4])) + '(H-H)' + ' '
                elif molecule_list.shape[1] == 9:
                    if molecules_reactants[j, 0] != 0:
                        react_string += 'C' + str(int(molecules_reactants[j, 0])) + ' '
                    if molecules_reactants[j, 1] != 0:
                        react_string += 'H' + str(int(molecules_reactants[j, 1])) + ' '
                    if molecules_reactants[j, 2] != 0:
                        react_string += 'O' + str(int(molecules_reactants[j, 2])) + ' '
                    if molecules_reactants[j, 3] != 0:
                        react_string += str(int(molecules_reactants[j, 3])) + '(C-C)' + ' '
                    if molecules_reactants[j, 4] != 0:
                        react_string += str(int(molecules_reactants[j, 4])) + '(C-H)' + ' '
                    if molecules_reactants[j, 5] != 0:
                        react_string += str(int(molecules_reactants[j, 5])) + '(C-O)' + ' '
                    if molecules_reactants[j, 6] != 0:
                        react_string += str(int(molecules_reactants[j, 6])) + '(H-H)' + ' '
                    if molecules_reactants[j, 7] != 0:
                        react_string += str(int(molecules_reactants[j, 7])) + '(H-O)' + ' '
                    if molecules_reactants[j, 8] != 0:
                        react_string += str(int(molecules_reactants[j, 8])) + '(O-O)' + ' '
                react_string += ' '
                if j != molecules_reactants.shape[0] - 1 or k != molecules_reactants[j, -1] - 1:
                    react_string += '+ '
        react_string += '=> '
        for j in range(molecules_products.shape[0]):
            for k in range(int(molecules_products[j, -1])):
                if molecule_list.shape[1] == 5:
                    if molecules_products[j, 0] != 0:
                        react_string += 'C' + str(int(molecules_products[j, 0])) + ' '
                    if molecules_products[j, 1] != 0:
                        react_string += 'H' + str(int(molecules_products[j, 1])) + ' '
                    if molecules_products[j, 2] != 0:
                        react_string += str(int(molecules_products[j, 2])) + '(C-C)' + ' '
                    if molecules_products[j, 3] != 0:
                        react_string += str(int(molecules_products[j, 3])) + '(C-H)' + ' '
                    if molecules_products[j, 4] != 0:
                        react_string += str(int(molecules_products[j, 4])) + '(H-H)' + ' '
                elif molecule_list.shape[1] == 9:
                    if molecules_products[j, 0] != 0:
                        react_string += 'C' + str(int(molecules_products[j, 0])) + ' '
                    if molecules_products[j, 1] != 0:
                        react_string += 'H' + str(int(molecules_products[j, 1])) + ' '
                    if molecules_products[j, 2] != 0:
                        react_string += 'O' + str(int(molecules_products[j, 2])) + ' '
                    if molecules_products[j, 3] != 0:
                        react_string += str(int(molecules_products[j, 3])) + '(C-C)' + ' '
                    if molecules_products[j, 4] != 0:
                        react_string += str(int(molecules_products[j, 4])) + '(C-H)' + ' '
                    if molecules_products[j, 5] != 0:
                        react_string += str(int(molecules_products[j, 5])) + '(C-O)' + ' '
                    if molecules_products[j, 6] != 0:
                        react_string += str(int(molecules_products[j, 6])) + '(H-H)' + ' '
                    if molecules_products[j, 7] != 0:
                        react_string += str(int(molecules_products[j, 7])) + '(H-O)' + ' '
                    if molecules_products[j, 8] != 0:
                        react_string += str(int(molecules_products[j, 8])) + '(O-O)' + ' '
                react_string += ' '
                if j != molecules_products.shape[0] - 1 or k != molecules_products[j, -1] - 1:
                    react_string += '+ '
        react_strings[i] = react_string
    return react_strings

def get_features_for_molecules_framework(reaction_dict):
    features_dict = {"Reactants": [], "Products": []}
    for i in range(reaction_dict.shape[0]):
        reactants_idx = np.where(reaction_dict[i] < 0)[0]
        products_idx = np.where(reaction_dict[i] > 0)[0]
        reactants_list = []
        products_list = []
        for j in range(reactants_idx.shape[0]):
            reactants_list.append([reactants_idx[j], abs(reaction_dict[i, reactants_idx[j]])])
        for j in range(products_idx.shape[0]):
            products_list.append([products_idx[j], reaction_dict[i, products_idx[j]]])
        features_dict["Reactants"].append(reactants_list)
        features_dict["Products"].append(products_list)
    df_features = pd.DataFrame(data=features_dict)
    return df_features