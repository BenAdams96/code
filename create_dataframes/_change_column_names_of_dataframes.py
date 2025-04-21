
from create_dataframes import _initial_dataframe, a_dataframes_descriptors_only,b_dataframes_reduced, c_dataframes_red_MD, \
    d_dataframes_MD_only, e_dataframes_DescPCA

from global_files import public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
import pandas as pd

def main():
    print('create dataframe files')

    column_mapping = {
        'Total': 'SASA',
        'num of H-bonds': 'H-bonds',
        'within_distance': 'H-bonds within 0.35A',
        'Mtot': 'Total dipole moment',
        'Bond': 'Ligand Bond energy',
        'U-B': 'Urey-Bradley energy',
        'Proper Dih.': 'Torsional energy',
        'Coul-SR:Other-Other': 'Coul-SR: Lig-Lig',
        'LJ-SR:Other-Other': 'LJ-SR: Lig-Lig',
        'Coul-14:Other-Other': 'Coul-14: Lig-Lig',
        'LJ-14:Other-Other': 'LJ-14: Lig-Lig',
        'Coul-SR:Other-SOL': 'Coul-SR: Lig-Sol',
        'LJ-SR:Other-SOL': 'LJ-SR: Lig-Sol',
        'RMSD (nm)': 'RMSD'
        # Add more mappings as needed
    }

    MD_output_df = pd.read_csv(pv.MD_outputfile_)
    MD_output_df = MD_output_df.rename(columns=column_mapping)
    MD_output_df.to_csv(pv.energyfolder_path_ / 'MD_output.csv', index=False)
if __name__ == "__main__":
    # Update public variables
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)
    # main()
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    main()
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # main()
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)

    # # Call main
    # main()



# %%
