
from create_dataframes import _initial_dataframe, a_dataframes_descriptors_only, b_dataframes_reduced, c_dataframes_red_MD, \
    d_dataframes_MD_only, e_dataframes_DescPCA, f_dataframes_DescPCA_MD, _change_column_names_of_dataframes, b_dataframes_reduced_redbefore, \
    e_dataframes_DescPCA_variance, f_dataframes_DescPCA_MD_variance


from global_files import public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

def main(include):
    print('create dataframe files')
    to_keep = ['potential','RMSD','epsilon','PSA','SASA','H-bonds', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','LJ-SR: Lig-Sol']
    variance = 0.98
    threshold = 0.85
    
    # _change_column_names_of_dataframes.main()
    # _initial_dataframe.main() #this creates the df with all (0.01 stepsize) which is initial_dataframe
    dfs_in_dict = a_dataframes_descriptors_only.main(include=include, write_out=True) #we only want dataframe of 1ns 2ns 3ns
    b_dataframes_reduced_redbefore.main(threshold=threshold, include = include, write_out=True)
    c_dataframes_red_MD.main(savefolder_name=pv.dfs_reduced_and_MD_path_, include = include, threshold=threshold, to_keep=to_keep)
    
    e_dataframes_DescPCA_variance.main(savefolder_name= pv.dfs_dPCA_var_path_, variance=variance, include = include, write_out=True)
    f_dataframes_DescPCA_MD_variance.main(savefolder_name=pv.dfs_dPCA_var_MD_path_, include = include, variance=variance, to_keep=to_keep)

    if DESCRIPTOR != 'GETAWAY':
        d_dataframes_MD_only.main(savefolder_name='MD only', include = include, to_keep=to_keep,write_out=True)

if __name__ == "__main__":
    # Update public variables
    include = [0,1,2,3,4,5,6,7,8,9,10,'c10','c20','c50'] #,'ta10c10' , 'CLt100_cl10_c10' #for timesteps, and for clustering
    
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.GETAWAY, protein_=DatasetProtein.GSK3)
    main(include)

    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.GETAWAY, protein_=DatasetProtein.GSK3)
    # main(include)

    # include = [0,1,2,3,4,5,6,7,8,9,10,'c10','c20','c50','c100','ta10c10','CLt50_cl10_c10'] #,'ta10c10' , 'CLt100_cl10_c10' #for timesteps, and for clustering
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    # main(include)

    # include = ['CLt50_cl10_c10'] #,'ta10c10' , 'CLt100_cl10_c10' #for timesteps, and for clustering
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # main(include)

    # include = [0,1,2,3,4,'c10','c20','c50','c100','ta4c10'] #,'ta10c10' , 'CLt100_cl10_c10' #for timesteps, and for clustering
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.CLK4)
    # main(include)

    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)
    # main()

    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)
    # main()
