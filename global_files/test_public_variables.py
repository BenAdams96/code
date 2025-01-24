
from pathlib import Path
# from global_files.public_variables_updated_Class import PublicVariables

# from global_files import public_variables
from global_files import public_variables as pv
from global_files.public_variables import PROTEIN, DESCRIPTOR, update_config  # Import from public_variables
from global_files.enums import Model_classic, Model_deep, Descriptor, DatasetProtein

# from global_files import public_variables_updated
# from global_files.public_variables_updated import DATASET_PROTEIN, DatasetProtein
# from global_files.public_variables_updated import DATASET_PROTEIN, DatasetProtein
# print(public_variables.dataset_protein_)
# print(public_variables_copy.dfs_descriptors_only_path_)

# public_variables.dataset_protein_ = 'GSK3'

# print(public_variables.dataset_protein_)
# print(public_variables_copy.dfs_descriptors_only_path_)


# print(public_variables_copy.DATASET_PROTEIN)
# print(public_variables_copy.dfs_descriptors_only_path_)

# public_variables_copy.DATASET_PROTEIN = 'GSK3'
# print(public_variables_copy.DATASET_PROTEIN)
# print(public_variables_copy.dfs_descriptors_only_path_)
# public_variables_copy.DATASET_PROTEIN = 'JAK1'
# print(public_variables_copy.DATASET_PROTEIN)
# print(public_variables_copy.dfs_descriptors_only_path_)


# public_variables_copy.update_config(new_dataset_protein='GSK3')
# print(public_variables_copy.DATASET_PROTEIN)
# print(public_variables_copy.dfs_descriptors_only_path_)

# print(public_variables_updated.DATASET_PROTEIN)
# public_variables_updated.DATASET_PROTEIN = 'GSK3'
# print(public_variables_updated.DATASET_PROTEIN)

# print(DATASET_PROTEIN)
# DATASET_PROTEIN = DatasetProtein.GSK3
# print(DATASET_PROTEIN)
# for protein in DatasetProtein:
#     DATASET_PROTEIN = protein
#     print(DATASET_PROTEIN)

# public_variables_updated.DATASET_PROTEIN = DatasetProtein.GSK3
# print(public_variables_updated.get_paths(DatasetProtein.JAK1))


# print(PublicVariables.DESCRIPTOR)

# print(PublicVariables.xvgfolder_path_)
# PublicVariables.DESCRIPTOR = PublicVariables.update_config(new_descriptor=PublicVariables.Descriptor.WHIM)
# print(PublicVariables.xvgfolder_path_)


# print(public_variables.PROTEIN)
# print(public_variables.get_variables())
# public_variables.update_config(Model_classic.XGB)
# print(public_variables.get_variables()) 
# print(public_variables.PROTEIN.dataset_length)
# print(public_variables.DESCRIPTOR.descriptor_length)
# print(PROTEIN)
# PROTEIN = DatasetProtein.GSK3
# print(PROTEIN)
# print(PROTEIN.dataset_length)
# print(public_variables.PROTEIN)
# update_config(new_dataset_protein=DatasetProtein.pparD)
# print(public_variables.MDsimulations_path_)
# print(public_variables.PROTEIN)

pv.update_config(model_=Model_classic.XGB, descriptor_=Descriptor.GETAWAY, protein_=DatasetProtein.JAK1)
print(pv.DESCRIPTOR)
print(pv.DESCRIPTOR.name)
print(pv.DESCRIPTOR.value)
print(pv.DESCRIPTOR.descriptor_length)


print(pv.ML_MODEL)
print(pv.ML_MODEL.name)
print(type(pv.ML_MODEL))
print(type(pv.ML_MODEL.name))
print(pv.ML_MODEL.model)
print(pv.ML_MODEL.hyperparameter_grid)


