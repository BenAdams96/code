# Project-specific imports
from global_files import public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

from create_dataframes import create_initial_dataframe,create_dataframes_descriptors_only, \
    create_dataframes_reduced,create_dataframes_reduced_MD,create_dataframes_MD_only, \
    create_dataframes_reduced_PCA, create_dataframes_reduced_MD_combined_PCA, create_dataframes_reduced_MD_PCA, \
    create_dataframes_all_PCA

def main():
    print('create dataframe files')

    # create_initial_dataframe.main() #this creates the df with all (0.01 stepsize) which is initial_dataframe
    create_dataframes_descriptors_only.main(time_interval= 1, timeinterval_list = [1,0.5,0.2,0.1]) #we only want dataframe of 1ns 2ns 3ns
    # create_dataframes_reduced.main(threshold=0.85)
    # create_dataframes_reduced_MD.main()
    # create_dataframes_MD_only.main()

    # create_dataframes_reduced_PCA.main()
    # create_dataframes_reduced_MD_PCA.main()
    # create_dataframes_reduced_MD_combined_PCA.main()
    # create_dataframes_all_PCA.main()
    
if __name__ == "__main__":
    # Update public variables
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)

    # Call main
    main()
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)

    # # Call main
    # main()



# %%
