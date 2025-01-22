# Project-specific imports
from global_files import public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

from create_dataframes import create_initial_dataframe,create_dataframes_descriptors_only, \
    create_dataframes_reduced,create_dataframes_reduced_MD,create_dataframes_MD_only

def main():
    print('create dataframe files')

    # screate_initial_dataframe.main()
    # create_dataframes_descriptors_only.main(time_interval= 1)
    # create_dataframes_reduced.main(threshold=0.85)
    # create_dataframes_reduced_MD.main()
    create_dataframes_MD_only.main()
    
if __name__ == "__main__":
    # Update public variables
    pv.update_config(model=Model_classic.RF, descriptor=Descriptor.WHIM, protein=DatasetProtein.JAK1)

    # Call main
    main()



# %%
