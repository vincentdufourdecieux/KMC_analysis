import numpy as np
import time

import initialization_KMC
import KMC_process
import save_data

seed = 1  # Random seed
num = 1  # Number of the KMC simulation
foldername_reactions = './Results_dump_in/'  # Folder path with the results of the simulations you want to use
                                                   # the reactions from
foldername_starting_molecules = './Results_dump_in_big_100ps/'  # Folder path with the results of the simulations you want
                                                            # to use for the starting molecules
foldername_save = './Results_dump_in_to_dump_in_big_100ps/'  # Folder path to save the results of the KMC

np.random.seed(seed)
t = time.process_time()

[reactions_KMC, atoms_KMC, molecules_KMC, end_t, reaction_rates, reaction_framework,
 first_frame] = initialization_KMC.get_all_for_KMC(foldername_reactions, foldername_starting_molecules)

KMC_run = KMC_process.KMC(end_t, reactions_KMC, atoms_KMC, reaction_rates)
print("Starting KMC", flush=True)
print(time.process_time(), flush=True)
[bond_change_KMC] = KMC_run.run_KMC(first_frame)
print("End KMC", flush=True)
print(time.process_time(), flush=True)
if reaction_framework == 'atomic':
    molecule_list = molecules_KMC.molecule_list
    timerange_KMC = molecules_KMC.time_range
    molecule_per_frame = molecules_KMC.molecule_all_frames[:timerange_KMC.shape[0], :molecule_list.shape[0]]
elif reaction_framework == 'molecular':
    timerange_KMC = reactions_KMC.time_range
    molecule_per_frame = reactions_KMC.molecule_per_frame[:timerange_KMC.shape[0], :]
    molecule_list = reactions_KMC.molecule_list

save_data.save_data_KMC(foldername_save, bond_change_KMC, molecule_per_frame, molecule_list, timerange_KMC, num)
