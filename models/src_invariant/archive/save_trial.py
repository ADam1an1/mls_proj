# save_trial.py

import numpy as np
from run_sim import * 
import h5py

def check_trial(file, trial_name, params, operators):
    """
    Returns true if a trial with specified
    name and params has been run and saved in
    the file specified
    """

    if trial_name in file:
        print("Checking keys")
        for key in file[trial_name].attrs:
            if isinstance(file[trial_name].attrs[key], np.ndarray):
                if not np.array_equal(file[trial_name].attrs[key], params[key]):
                    print(key, file[trial_name].attrs[key], params[key])
                    print("Trial inconsistencies, overwriting previous trial")
                    return False
            else:
                if not file[trial_name].attrs[key] == params[key]:
                    print(key, file[trial_name].attrs[key], params[key])
                    print("Trial inconsistencies, overwriting previous trial")
                    return False

        for op in operators:
            if not op in file[trial_name]:
                print("Missing operator data")
                return False
    else:
        print("Trial not previously run")
        return False

    print("Found data")
    return True

def get_trial_data(file_path, trial_names, trial_data):
    """
    Retrieves the specified data type from a file
    for all trials that are named
    **note that the trials are not checked for params** 
    """

    data = []
    with h5py.File(file_path, 'r') as f:
        for trial in trial_names:
            data.append(f[trial + "/" + trial_data][:])
    return data

def run_trial(file_path, trial_name, params, operators):
    """
    Function to run trial with saving and opening
    functionality
    """

    with h5py.File(file_path, 'a') as f:
        # run trial as specified 
        if not check_trial(f, trial_name, params, operators):
            # overwrite old trial if diff params
            if trial_name in f:
                del f[trial_name]
            trial_group = f.create_group(trial_name)
            result = run_simulation(system_e_levels=params['system_e_levels'],
                                    photon_freqs=params['photon_freqs'],
                                    max_photon_nums=params['photon_max_nums'],
                                    couplings_dict=params['couplings'],
                                    system_starts=params['system_starts'],
                                    photon_starts=params['photon_starts'],
                                    time=params['time'], steps=params['steps'],
                                    spatial=params['spatial'], track=operators, 
                                   model=params['model'])

            for key, value in params.items():
                trial_group.attrs[key] = value

            shift = 0

            if "energy" in operators:
                trial_group.create_dataset("energy", data_result.expect[0])
                shift += 1

            if "photons" in operators: 
                trial_group.create_dataset("photons", data=result.expect[shift:len(params['photon_freqs']) + shift])
                shift += len(params['photon_freqs'])

            if "states" in operators:
                trial_group.create_dataset("states", data=result.expect[shift:])

        print("Trial completed")


