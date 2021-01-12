# KMC_analysis
Allows to construct of a chemical kinetics model from an .atom file of an atomistic simulation and run a Kinetic Monte Carlo simulation with this kinetics model.

For any question, please email at vdufourd@stanford.edu.

# Step 1: loading the required packages and files
Work with Python 3.8.
Load the packages in the requirements.txt file.
Load all the .py file of this repository (with git clone for example)

# Step 2: analyzing the atomistic simulation analysis
Open the file "MD_analysis.py", enter the path to your .atom file, the path of the folder you want to save the results in, the framework ("atomic" or "molecular") 
you want to use, the start and end time of the atomistic simulation, the start and end time of the analysis, the bond duration and bond length criterion and
the timestep between two frames of the atomistic simulation.

Run 'python3 MD_analysis.py'

You will obtain several output files:
- Reactions.csv : a .csv that contains the description of the different reactions that occurred in the simulation, the reaction rates, types, number of occurrences,
number of times the reaction could have happened.
- First_frame.npz: a .npz file containing a Scipy sparse matrix of shape (num of atoms, num of atoms) containing the adjacency matrix of the initial state of your
system.
- Molecules_per_frame.npz: a .npz file containing a Scipy sparse matrix of shape (num of frames with a reaction, num of unique molecules) containing the number of each molecule at each time in the atomistic simulation
- Molecule_list.txt : a .txt file containing a NumPy array with "num of unique molecules" row, where each row is the description of the molecule. 
- Time_range.txt : a .txt file containing a NumPy array with "num of frames with a reaction" row, where each row is the time in picoseconds corresponding to this frame.
- Bond_change.pkl : a .pkl file containing a dictionary. The key of the dictionary are the frame at which the reactions in the values occurred
- Parameters.pkl : a .pkl file containing a dictionary of several useful parameters.

# Step 3: running the Kinetic Monte Carlo simulation
Open the file "KMC_simulation.py", enter the path of the folders containing the results of the analyzes of the atomistic simulation analysis of the system you want to use the kinetics model, the system you want to try to predict and the folder where you want to save the files.

Run 'python3 KMC_simulation.py"

You will obtain several output files:
- Molecules_per_frame_KMC_{num}.npz: a .npz file containing a Scipy sparse matrix of shape (num of frames with a reaction, num of unique molecules) containing the number of each molecule at each time in the KMC simulation.
- Molecule_list_KMC_{num}.txt : a .txt file containing a NumPy array with "num of unique molecules" row, where each row is the description of the molecule in the KMC simulation. 
- Time_range_KMC_{num}.txt : a .txt file containing a NumPy array with "num of frames with a reaction" row, where each row is the time in picoseconds corresponding to this frame in the KMC simulation.
