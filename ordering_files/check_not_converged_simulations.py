

from global_files import csv_to_dictionary,global_functions, public_variables as pv
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from pathlib import Path
import shutil
import os

def check_if_converged(MD_path):
    not_converged_molecules = []
    # Loop over the molecules (using padded numbers)
    for padded_num in (f"{i:03}" for i in range(1, pv.PROTEIN.dataset_length+1)): #pv.PROTEIN.dataset_length
        combined_path = MD_path / padded_num
        print(combined_path)
        if combined_path.exists() and combined_path.is_dir():
            os.chdir(MD_path / f'{padded_num}')
            for file in combined_path.iterdir():  # Iterate over files in the directory
                if file.name.endswith("cg.log"):
                    with open(file, 'r') as f:
                        for line in f:
                            if 'converged to Fmax' in line:
                                print(f"✓ Converged in {file.name}")
                                break
                        else:
                            not_converged_molecules.append(padded_num)
                if file.name.endswith("steep.log"):
                    with open(file, 'r') as f:
                        for line in f:
                            if 'converged to Fmax' in line:
                                print(f"✓ Converged in {file.name}")
                                break
                        else:
                            not_converged_molecules.append(int(padded_num))


    return not_converged_molecules

def check_minfile_but_no_prodfile(MD_path):
    seg_error_molecules = []
    invalid_molecules = []
    for padded_num in (f"{i:03}" for i in range(1, pv.PROTEIN.dataset_length+1)): #pv.PROTEIN.dataset_length
        combined_path = MD_path / padded_num
        print(combined_path)
        if combined_path.exists() and combined_path.is_dir():
            os.chdir(MD_path / f'{padded_num}')
            has_min = False
            has_prod = False

            for file in combined_path.iterdir():
                filename = file.name.lower()
                if 'min' in filename:
                    has_min = True
                if 'prod' in filename:
                    has_prod = True

            if has_min and not has_prod:
                seg_error_molecules.append(padded_num)
            elif not has_min and not has_prod:
                invalid_molecules.append(padded_num)
    return seg_error_molecules, invalid_molecules

def check_which_0ns(MD_path):
    one_ns_sim = []
    ten_ns_sim = []
    for padded_num in (f"{i:03}" for i in range(1, pv.PROTEIN.dataset_length+1)): #pv.PROTEIN.dataset_length
        combined_path = MD_path / padded_num
        print(combined_path)
        if combined_path.exists() and combined_path.is_dir():
            os.chdir(MD_path / f'{padded_num}')
            
            log_file = combined_path / f"{padded_num}_prod.log"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    for line in f:
                        if line.strip().startswith("nsteps"):
                            nsteps = int(line.strip().split('=')[1].strip())
                            if nsteps == 5000000:
                                ten_ns_sim.append(padded_num)

                            else:
                                one_ns_sim.append(padded_num)
                            break
    return one_ns_sim, ten_ns_sim

def print_missing_file(MD_path):
    for padded_num in (f"{i:03}" for i in range(1, pv.PROTEIN.dataset_length+1)): #pv.PROTEIN.dataset_length
        combined_path = MD_path / padded_num
        if combined_path.exists() and combined_path.is_dir():
            continue
        else:
            print(f'{padded_num} doesnt exist')
    return

def stopped_early(MD_path):
    mols_stopped_early = []
    for padded_num in (f"{i:03}" for i in range(1, pv.PROTEIN.dataset_length+1)): #pv.PROTEIN.dataset_length
        combined_path = MD_path / padded_num
        if combined_path.exists() and combined_path.is_dir():
            os.chdir(MD_path / f'{padded_num}')
            prod_log_file = combined_path / f"{padded_num}_prod.log"
            if prod_log_file.exists():
                with open(prod_log_file, 'r') as f:
                    for line in f:
                        if 'stopping within' in line:
                            print(padded_num)
                            mols_stopped_early.append(padded_num)
                            break
    return mols_stopped_early

def main():
    # delete_hash(Path("/home/ben/Download/runMDsimulations_CLK4/alloutputsFinal"))
    # delete_hash(pv.MDsimulations_path_)
    mols_stopped_early = stopped_early(pv.MDsimulations_path_)
    print(mols_stopped_early)
    one_ns_sim, ten_ns_sim = check_which_0ns(pv.MDsimulations_path_)
    print('one ns')
    print(one_ns_sim)
    print('10 ns')
    print(ten_ns_sim)
    
    seg_error_molecules, invalid_molecules = check_minfile_but_no_prodfile(pv.MDsimulations_path_)
    not_converged_molecules = check_if_converged(pv.MDsimulations_path_)
    molecules_list, valid_mols, invalid_mols = global_functions.get_molecules_lists(pv.MDsimulations_path_)
    print_missing_file(pv.MDsimulations_path_)
    print('not converged')
    print(not_converged_molecules)
    print('seg error molecules')
    print(seg_error_molecules)
    print('invalid molecules')
    print(invalid_molecules)
    print(molecules_list, valid_mols, invalid_mols, len(invalid_mols))
    return

if __name__ == "__main__":
    pv.update_config(protein_=DatasetProtein.pparD)
    main()
    # pv.update_config(model_= Model_classic.RF, protein_=DatasetProtein.GSK3)
    # main()
    # pv.update_config(model_= Model_classic.RF, protein_=DatasetProtein.pparD)
    # main()

    