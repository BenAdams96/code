
from create_dataframes import _initial_dataframe, a_dataframes_descriptors_only,b_dataframes_reduced, c_dataframes_red_MD, \
    d_dataframes_MD, e_dataframes_DescPCA, f_dataframes_MDPCA, g_dataframes_DescPCA_MDPCA, h_dataframes_DescMDPCA, i_dataframes_MD_only_changed, \
    j_dataframes_red_MD_changed, g_dataframes_DescPCA_MDPCA_minPC1, k_dataframes_red_MDPCA, _change_column_names_of_dataframes, i_dataframes_MD_only_changed_improvedq

from global_files import public_variables as pv
from global_files.public_variables import ML_MODEL, PROTEIN, DESCRIPTOR
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

def main():
    print('create dataframe files')
    # _change_column_names_of_dataframes.main()
    # _initial_dataframe.main() #this creates the df with all (0.01 stepsize) which is initial_dataframe
    # a_dataframes_descriptors_only.main(time_interval= 1, timeinterval_list = [1,0.5,0.2,0.1]) #we only want dataframe of 1ns 2ns 3ns
    # b_dataframes_reduced.main(threshold=0.85)
    ########### c_dataframes_red_MD.main()
    ########### d_dataframes_MD.main()

    # ################ red + MD (different ones)
    # j_dataframes_red_MD_changed.main(savefolder_name='red MD_old', to_keep=['rmsd','Gyration','epsilon','PSA','SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy'])
    # j_dataframes_red_MD_changed.main(savefolder_name='red MD_new', to_keep=['rmsd','Gyration','epsilon','PSA','SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','LJ-SR: Lig-Sol'])
    # j_dataframes_red_MD_changed.main(savefolder_name='red MD_new reduced', to_keep=['PSA','SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','LJ-SR: Lig-Sol'])

    

    # j_dataframes_red_MD_changed.main(savefolder_name='red MD_old', to_keep=['rmsd','Gyration','epsilon','PSA','SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy'])
    # j_dataframes_red_MD_changed.main(savefolder_name='red MD_new', to_keep=['rmsd','Gyration','epsilon','PSA','SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','LJ-SR: Lig-Sol'])
    # j_dataframes_red_MD_changed.main(savefolder_name='red MD_new reduced', to_keep=['PSA','SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','LJ-SR: Lig-Sol'])

    # ################MD different ones
    # i_dataframes_MD_only_changed.main(savefolder_name='MD_old only', to_keep=['rmsd','Gyration','epsilon','PSA','SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy'])
    # i_dataframes_MD_only_changed.main(savefolder_name='MD_old only reduced', to_keep=['PSA','SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy'])
    # i_dataframes_MD_only_changed.main(savefolder_name='MD_new only', to_keep=['rmsd','Gyration','epsilon','PSA','SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','LJ-SR: Lig-Sol'])
    # i_dataframes_MD_only_changed.main(savefolder_name='MD_new only reduced', to_keep=['PSA','SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','LJ-SR: Lig-Sol'])
    i_dataframes_MD_only_changed_improvedq.main(savefolder_name='MD_new onlyall', to_keep=['rmsd','Gyration','epsilon','Schlitter Entropy','Quasiharmonic Entropy','PSA','SASA','num of H-bonds','H-bonds within 0.35A', 'Total dipole moment', 'Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','LJ-SR: Lig-Sol'])
    i_dataframes_MD_only_changed_improvedq.main(savefolder_name='MD_new only3', to_keep=['Schlitter Entropy','Quasiharmonic Entropy','PSA','SASA','num of H-bonds','H-bonds within 0.35A','Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','LJ-14: Lig-Lig','Coul-SR: Lig-Sol','LJ-SR: Lig-Sol'])
    i_dataframes_MD_only_changed_improvedq.main(savefolder_name='MD_new only4', to_keep=['SASA','Ligand Bond energy', 'Urey-Bradley energy', 'Torsional energy', 'Coul-SR: Lig-Lig','LJ-SR: Lig-Lig','Coul-14: Lig-Lig','Coul-SR: Lig-Sol'])

    # ###############PCA of descriptors only
    e_dataframes_DescPCA.main(components=10)
    # e_dataframes_DescPCA.main(components=20)

    # f_dataframes_MDPCA.main()

    # ###############combine Descriptors and MD_new -> do PC of total
    # h_dataframes_DescMDPCA.main(components=10)
    # h_dataframes_DescMDPCA.main(components=20)


    # ################ descPCA20 + MD variants
    # ############### g_dataframes_DescPCA_MDPCA.main(savefolder_name='DescPCA20 MDold', dfs_MD_path='MD_old only') dont work
    # ################ g_dataframes_DescPCA_MDPCA.main(savefolder_name='DescPCA20 MDnew', dfs_MD_path='MD_new only')
    # ################ g_dataframes_DescPCA_MDPCA.main(savefolder_name='DescPCA20 MDnew red', dfs_MD_path='MD_new only reduced')
    # g_dataframes_DescPCA_MDPCA.main(savefolder_name='DescPCA20 MDnewPCA', dfs_MD_path='MDnewPCA')

    # # descPCA20 + MDnew - PC1
    # g_dataframes_DescPCA_MDPCA_minPC1.main(savefolder_name='DescPCA20 MDnewPCA minus PC1', dfs_MD_path='MDnewPCA', minus='PCA_1')
    # g_dataframes_DescPCA_MDPCA_minPC1.main(savefolder_name='DescPCA20 MDnewPCA minus PCMD1', dfs_MD_path='MDnewPCA', minus='PCA_1_MD')


   ############### # reduced + MDPCA
    ###############k_dataframes_red_MDPCA.main(savefolder_name='red MDnewPCA')



if __name__ == "__main__":
    # Update public variables
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.JAK1)

    # Call main
    main()
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.GSK3)

    # Call main
    main()
    pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)

    # Call main
    main()
    
    
    # pv.update_config(model_=Model_classic.RF, descriptor_=Descriptor.WHIM, protein_=DatasetProtein.pparD)

    # # Call main
    # main()



# %%
