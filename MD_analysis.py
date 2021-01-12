# Standard library imports
import numpy as np
import time

# Local application imports
import initialization
import bond_analysis
import reactions_atomic
import reactions_molecular
import atoms
import molecule_creation
import save_data

atom_file = 'dump_in_big.atom'  # Simulation file path. MUST BE AT THE .ATOM FORMAT
folder_name_save = './Results_dump_in_big_molecular/'  # Folder path to save the results
reaction_framework = 'molecular'  # 'atomic' to use atomic features, 'molecular' to use molecular features
start_MD_time = 0  # In ps
end_MD_time = 100  # In ps
start_analysis_time = 0  # In ps, needs to be bigger than start_MD_time
end_analysis_time = 100  # In ps, needs to be smaller than end_MD_time
bond_duration_value = 8  # Number of frames for the bond duration criterion
timestep = 0.012  # In ps, timestep between two frames of the simulation
bond_length_criterion = [1.98, 1.57, 1.09]  # In Angstroms, different bond length criterion, must be in the order
                                            # [Type 1 - Type 1, Type 1 - Type 2, Type 2 - Type 2]

t = time.process_time()

# Obtain initial data
[pipeline, data, num_of_atoms, num_of_frames, num_of_type] = initialization.initialize_system(atom_file)

# Start and end analysis at these frames
start_frame_MD = int(start_MD_time / timestep)
if int(end_MD_time / timestep) > num_of_frames:
    end_frame_MD = num_of_frames
else:
    end_frame_MD = int(end_MD_time / timestep)
start_frame_analysis = int(start_analysis_time / timestep)
if int(end_analysis_time / timestep) > num_of_frames:
    end_frame_analysis = num_of_frames
else:
    end_frame_analysis = int(end_analysis_time / timestep)

# Define the bond length criterion and apply it to the data
initialization.set_bond_length(bond_length_criterion, pipeline, num_of_type)

# Get the types of the atoms
atom_types = initialization.get_atom_types(data, num_of_atoms)

# Initialize the "atoms", "molecules" and "reactions" classes
atoms = atoms.Atoms(num_of_type, atom_types)
molecules = molecule_creation.MoleculeList(num_of_atoms, np.array([]), atom_types, atoms, num_of_type, start_frame_MD,
                                           end_frame_MD)
if reaction_framework == 'atomic':
    reactions = reactions_atomic.Reactions(num_of_type, np.array([]), np.array([]), atoms, 0, molecules)
elif reaction_framework == 'molecular':
    reactions = reactions_molecular.Reactions(np.array([]), molecules, np.array([]))

# Run the bond analysis, you will now have the first_frame adjacency matrix, the evolution of the bond,
# the reactions that happened and the number of times each one happened.
print("Beginning bond analysis", flush=True)
print(time.process_time(), flush=True)
bond_analysis = bond_analysis.BondAnalysis(pipeline, bond_duration_value, num_of_atoms)
[first_frame, bond_change, first_frame_analysis] = bond_analysis.run_analysis(reactions, molecules, start_frame_MD,
                                                                              end_frame_MD, start_frame_analysis,
                                                                              end_frame_analysis)
print(time.process_time(), flush=True)

# Symmetrize first_frame arrays
first_frame = (first_frame.T + first_frame).toarray()
first_frame_analysis = (first_frame_analysis.T + first_frame_analysis).toarray()
if reaction_framework == 'molecular':
    reactions.truncate_reaction_dict()

molecule_list = molecules.molecule_list
timerange = molecules.time_range
molecule_per_frame = molecules.molecule_all_frames[:timerange.shape[0], :molecule_list.shape[0]]

# Go through all the time steps to know the number of times each reaction could have happened
print("Beginning get h", flush=True)
print(time.process_time(), flush=True)
if reaction_framework == 'atomic':
    [h, h_per_frame] = reactions.get_h(first_frame_analysis, bond_change, start_frame_analysis, end_frame_analysis)
elif reaction_framework == 'molecular':
    [h, h_per_frame] = reactions.get_h(molecule_per_frame, timerange, start_frame_analysis, end_frame_analysis)
reaction_rates = np.zeros([h.shape[0]])
for i in range(h.shape[0]):
    if h[i] != 0:
        reaction_rates[i] = reactions.reaction_occurrences[i] / (timestep * h[i])

print(time.process_time(), flush=True)
if reaction_framework == 'atomic':
    save_data.save_MD_data_atomic(folder_name_save, reaction_rates, reactions.reaction_type, reactions.reaction_occurrences,
                             h, reactions.reaction_dict, start_frame_MD, end_frame_MD,
                             num_of_type, num_of_atoms, atom_types, timestep, reaction_framework, bond_change,
                             first_frame, molecule_per_frame, molecule_list, timerange)
elif reaction_framework == 'molecular':
    save_data.save_MD_data_molecular(folder_name_save, reaction_rates, reactions.reaction_occurrences, h,
                             reactions.reaction_dict, molecule_list, start_frame_MD,
                             end_frame_MD, num_of_type, num_of_atoms, atom_types, timestep, reaction_framework,
                             bond_change, first_frame, molecule_per_frame, timerange)
print(time.process_time(), flush=True)
