# public_variables.py

from pathlib import Path
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein
from typing import Union

# Initialize basic paths
base_path_ = Path(__file__).resolve().parent.parent.parent  # 'Afstuderen' folder
Afstuderen_path_ = base_path_
code_path_ = base_path_ / 'code'

# Initialize dataset-related variables
ML_MODEL: Union[Model_classic, Model_deep] = Model_classic.RF
DESCRIPTOR: Descriptor = Descriptor.WHIM
PROTEIN: DatasetProtein = DatasetProtein.JAK1
correlation_threshold_ = 0.85
MDfeatures = ["Bond","U-B","Proper-Dih.","Coul-SR:Other-Other","LJ-SR:Other-Other","Coul-14:Other-Other","LJ-14:Other-Other","Coul-SR:Other-SOL","LJ-SR:Other-SOL"]
# rerun_MD_features = ["Coul-SR:Other-Other","LJ-SR:Other-Other","Coul-14:Other-Other","LJ-14:Other-Other","Coul-SR:Other-SOL","LJ-SR:Other-SOL"]

# Function to update paths dynamically
def update_paths():
    global dataset_path_, dataframes_master_, initial_dataframe_, dfs_descriptors_only_path_, dfs_reduced_path_, dfs_reduced_and_MD_path_, \
        dfs_MD_only_path_, Modelresults_folder_, Modelresults_combined_folder_, MDsimulations_folder_, true_predicted , ligand_conformations_system_path_,\
            MDsimulations_path_, ligand_conformations_folder_, ligand_conformations_path_, energyfolder_path_, Modelresults_plots, \
            edrfolder_path_, xvgfolder_path_, MD_outputfile_, MDfeatures_allmol_csvfile, Inner_train_Val_losses, Outer_train_Val_losses, \
            dfs_reduced_PCA_path_, dfs_reduced_MD_PCA_path_, dfs_reduced_and_MD_combined_PCA_path_, dfs_all_PCA, dfs_2D_path

    #Protein dataset
    dataset_path_ = base_path_ / 'dataZ/datasets' / f"{PROTEIN}_dataset.csv"
    
    # Dynamically update paths
    dataframes_master_ = Afstuderen_path_ / "dataframes" / f'dataframes_{PROTEIN}_{DESCRIPTOR}'
    initial_dataframe_ = dataframes_master_ / Path('initial_dataframe.csv')
    dfs_descriptors_only_path_ = dataframes_master_ / 'descriptors only'
    dfs_reduced_path_ = dataframes_master_ / f'reduced_t{correlation_threshold_}'
    dfs_reduced_PCA_path_ = dataframes_master_ / f'reduced_desc_PCA10'
    dfs_reduced_MD_PCA_path_ = dataframes_master_ / f'reduced_MD_PCA10'
    dfs_reduced_and_MD_combined_PCA_path_ = dataframes_master_ / f'combined_reduced_MD_PCA10'
    dfs_all_PCA = dataframes_master_ / f'all_PCA'
    dfs_reduced_and_MD_path_ = dataframes_master_ / f'reduced_t{correlation_threshold_}_MD'
    dfs_MD_only_path_ = dataframes_master_ / 'MD only'
    dfs_2D_path = dataframes_master_ / '2D'

    Modelresults_folder_ = Path(f'ModelResults_{ML_MODEL}') #not a path because can be in different paths
    Modelresults_combined_folder_ = f'ModelResults_combined_{ML_MODEL}'
    true_predicted = Modelresults_folder_ / 'true_predicted'
    Inner_train_Val_losses =  Modelresults_folder_ / 'inner_train_val_losses'
    Outer_train_Val_losses =  Modelresults_folder_ / 'outer_train_val_losses'
    Modelresults_plots = Modelresults_folder_ / 'plots'
    
    #MD simulations folder file
    MDsimulations_folder_ = f'MDsimulations_{PROTEIN}'
    MDsimulations_path_ = Afstuderen_path_ / "MDsimulations" / MDsimulations_folder_

    ligand_conformations_folder_ = f'ligand_conformations_{PROTEIN}'
    ligand_conformations_system_folder_ = f'ligand_conformations_system_{PROTEIN}'
    ligand_conformations_path_ = Afstuderen_path_ / "dataZ/ligand_conformation_files" / ligand_conformations_folder_
    ligand_conformations_system_path_ = Afstuderen_path_ / "dataZ/ligand_conformation_files" / ligand_conformations_system_folder_


    energyfolder_path_ = Afstuderen_path_ / "dataZ/MD_energy_files" / f'energyfolder_files_{PROTEIN}'
    edrfolder_path_ = energyfolder_path_ / f'edr_files_{PROTEIN}'
    xvgfolder_path_ = energyfolder_path_ / f'xvg_files_MD_features_{PROTEIN}'
    MD_outputfile_ = energyfolder_path_ / 'MD_output.csv'
    MDfeatures_allmol_csvfile = f'MD_features_{PROTEIN}.csv'


# Call update_paths once to initialize the paths
update_paths()

# Config update function for dynamically changing variables
def update_config(model_: Union[Model_classic, Model_deep]=None,
                  descriptor_: Descriptor=None,
                  protein_: DatasetProtein = None):
    global ML_MODEL, DESCRIPTOR, PROTEIN
    if model_:
        ML_MODEL = model_
    if descriptor_:
        DESCRIPTOR = descriptor_
    if protein_:
        PROTEIN = protein_
    # Recalculate dependent paths when dataset or descriptor changes
    update_paths()

def get_paths(model_: Union[Model_classic, Model_deep]=None,
                  descriptor_: Descriptor=None,
                  protein_: DatasetProtein = None):
    update_config(model_=model_, descriptor_=descriptor_, protein_=protein_)
    return [dfs_descriptors_only_path_, dfs_reduced_path_, dfs_reduced_and_MD_path_, dfs_MD_only_path_]

def get_variables():
    return ML_MODEL, DESCRIPTOR, PROTEIN


# dataframes_folder_red_ = f'dataframes_{dataset_protein_}_{descriptors_}_i{timeinterval_snapshots}_t{correlation_threshold_}'
# dataframes_folder_red_MD = f'dataframes_{dataset_protein_}_{descriptors_}_i{timeinterval_snapshots}_t{correlation_threshold_}_MD'
# Modelresults_ = f'ModelResults_{model_}_{dataset_protein_}_{descriptors_}_i{timeinterval_snapshots}'

# dataframes_folder = 'dataframesWHIMJAK1_with0.95'
# dataframes_folder_red = 'dataframesWHIMJAK1_reduced_with0.95'
# dataframes_folder_red_MD = 'dataframesWHIMJAK1_red_MD_with0.95_good'
# Modelresults_RF = 'ModelResults_RF_095'

#NOTE: features is the one that is used. for prepare_energy_files.py
featuresold = ["Bond","U-B","Proper-Dih.","Improper-Dih.","LJ-(SR)","Coulomb-(SR)","Potential","Total-Energy", "Enthalpy"]
features = ["Bond","U-B","Proper-Dih."]
#NOTE: options:
    # 1  Bond             2  U-B              3  Proper-Dih.      4  Improper-Dih. 
    # 5  LJ-14            6  Coulomb-14       7  LJ-(SR)          8  Disper.-corr. 
    # 9  Coulomb-(SR)    10  Coul.-recip.    11  Potential       12  Kinetic-En.   
    # 13  Total-Energy    14  Conserved-En.   15  Temperature     16  Pres.-DC      
    # 17  Pressure        18  Constr.-rmsd    19  Box-X           20  Box-Y         
    # 21  Box-Z           22  Volume          23  Density         24  pV            
    # 25  Enthalpy        26  Vir-XX          27  Vir-XY          28  Vir-XZ        
    # 29  Vir-YX          30  Vir-YY          31  Vir-YZ          32  Vir-ZX        
    # 33  Vir-ZY          34  Vir-ZZ          35  Pres-XX         36  Pres-XY       
    # 37  Pres-XZ         38  Pres-YX         39  Pres-YY         40  Pres-YZ       
    # 41  Pres-ZX         42  Pres-ZY         43  Pres-ZZ         44  #Surf*SurfTen 
    # 45  T-Other         46  T-SOL
    #NOTE: not every molecule contains IMPROPER-DIH!
# features_all = features = [
#     'Bond', 'U-B', 'Proper-Dih.', 
#     'LJ-14', 'Coulomb-14', 'LJ-(SR)', 'Disper.-corr.', 
#     'Coulomb-(SR)', 'Coul.-recip.', 'Potential', 'Kinetic-En.', 
#     'Total-Energy', 'Conserved-En.', 'Temperature', 'Pres.-DC', 
#     'Pressure', 'Constr.-rmsd', 'Box-X', 'Box-Y', 
#     'Box-Z', 'Volume', 'Density', 'pV', 
#     'Enthalpy', 'Vir-XX', 'Vir-XY', 'Vir-XZ', 
#     'Vir-YX', 'Vir-YY', 'Vir-YZ', 'Vir-ZX', 
#     'Vir-ZY', 'Vir-ZZ', 'Pres-XX', 'Pres-XY', 
#     'Pres-XZ', 'Pres-YX', 'Pres-YY', 'Pres-YZ', 
#     'Pres-ZX', 'Pres-ZY', 'Pres-ZZ', '#Surf*SurfTen', 
#     'T-Other', 'T-SOL']


# dataframes_folder = 'dataframesWHIMJAK1'
# dataframes_folder_red = 'dataframesWHIMJAK1_reduced'
# dataframes_folder_red_MD = 'dataframesWHIMJAK1_red_MD'
# Modelresults_RF = 'ModelResults_RF'

reduced_models_to_create_ = {
        'feature_importance': [0.03],
        'x_features': [15,20,25]
        }

parameter_grid_ = {
        'kfold_': [10],
        'scoring_': [('r2','R-squared (R²)')] # ('neg_root_mean_squared_error','RMSE'),
        }

# parameter_grid = {
    #         'kfold_': [5,10],
    #         'scoring_': [('neg_root_mean_squared_error','RMSE'),('r2','R-squared (R²)')],
    #         }

hyperparameter_grid_ = {
            'n_estimators': [100],
            'max_depth': [3,8],
            'min_samples_split': [3,8],
            'min_samples_leaf': [5],
            'max_features': ['sqrt'] #'None' can lead to overfitting.
        }
# hyperparameter_grid_RF = {
#     'n_estimators': [100, 200],
#     'max_depth': [8, 15],
#     'min_samples_split': [2, 10],
#     'min_samples_leaf': [2, 5],
#     'max_features': ['sqrt']
# }
hyperparameter_grid_RF = {
    'n_estimators': [100],
    'max_depth': [5,8],
    'min_samples_split': [2,5],
    'min_samples_leaf': [2,5],
    'max_features': ['sqrt']
}
# hyperparameter_grid_ = {
#             'n_estimators': [100],
#             'max_depth': [10],
#             'min_samples_split': [10],
#             'min_samples_leaf': [5],
#             'max_features': ['sqrt'] #'None' can lead to overfitting.
#         }

# hyperparameter_grid_ = {
#             'n_estimators': [100,150,300],
#             'max_depth': [10, 30, 50],
#             'min_samples_split': [10, 25, 4],
#             'min_samples_leaf': [5, 10, 15],
#             'max_features': ['sqrt'] #'None' can lead to overfitting.
#         }

hyperparameter_grid_XGB = {
    'n_estimators': [100,150],          # Number of trees (lower values for quicker training)
    'max_depth': [2,5],            # Maximum depth of each tree (shallower trees to avoid overfitting)
    'learning_rate': [0.2],       # Learning rate (smaller values for more gradual training)
    'subsample': [0.7],            # Subsample ratio of the training instance (to prevent overfitting)
    'colsample_bytree': [0.7],     # Subsample ratio of columns when constructing each tree
    'gamma': [0.1],                  # Minimum loss reduction required to make a further partition on a leaf node
}

# hyperparameter_grid_SVM = {
#     'kernel': ['rbf'],        # Most commonly effective kernels
#     'C': [0.1],                      # Reasonable regularization strengths for regression
#     'epsilon': [0.01],             # Standard values for error tolerance
#     'gamma': ['scale']             # Default and a specific small value for tuning influence
# }

hyperparameter_grid_SVM = {
    'kernel': ['rbf'],  # A non-linear and linear option
    'C': [0.1, 1],               # Two reasonable regularization strengths
    'epsilon': [0.1, 0.5],       # Moderate tolerances for prediction errors
    'gamma': ['scale', 'auto'],          # Default gamma for simplicity
}

hyperparameter_grid_ENR = {
    'alpha': [0.1, 1.0, 10.0],
    'l1_ratio': [0.2, 0.5, 0.8]
}

# hyperparameter_grid_XGboost = {
#     'n_estimators': [50, 100],          # Number of trees (lower values for quicker training)
#     'max_depth': [3, 5, 7],             # Maximum depth of each tree (shallower trees to avoid overfitting)
#     'learning_rate': [0.01, 0.1],       # Learning rate (smaller values for more gradual training)
#     'subsample': [0.6, 0.8],            # Subsample ratio of the training instance (to prevent overfitting)
#     'colsample_bytree': [0.6, 0.8],     # Subsample ratio of columns when constructing each tree
#     'gamma': [0, 0.1],                  # Minimum loss reduction required to make a further partition on a leaf node
# }




LSTM_master_ = base_path_ / Path(f'LSTM folder')





