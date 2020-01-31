import numpy as np
from rdkit import Chem
from sklearn.model_selection import train_test_split
from feature import *

def makeData(proteinName):
    # load data =========================================
    print('start loading train data')
    afile = 'TOX21/' + proteinName + '_wholetraining.smiles'
    smi = Chem.SmilesMolSupplier(afile, delimiter=' ', titleLine=False)  # smi var will not be used afterwards
    mols = [mol for mol in smi if mol is not None]

    # Make Feature Matrix ===============================
    F_list, T_list = [], []
    for mol in mols:
        if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > 400:
            print("too long mol was ignored")
        else:
            F_list.append(mol_to_feature(mol, -1, 400))
            T_list.append(mol.GetProp('_Name'))

    # Setting Dataset to model ==========================
    for i in range(len(F_list[0])):
        print(F_list[0][i])
    F_list = np.asarray(F_list, dtype=np.float32).reshape(400, 42)  # 42 is lensize
    for i in range(len(F_list[0])):
        print(F_list[0][i])

makeData("NR-AR")