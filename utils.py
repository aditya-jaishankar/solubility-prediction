#%%
import numpy as np
import pandas as pd

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps

import glob
import imageio
from tqdm import tqdm

import matplotlib.pyplot as plt
# matplotlib.use('agg')

import torch

import os


#%%

# We first process the data into tuples: the first element is the torch tensor
# for the image, and the second is the hydrophobicity/hydrophilicity label. 
# Note that the solubility units are given in lopS (log of molar solubility)

dir_path = os.path.dirname(os.path.realpath(__file__))
data = pd.read_csv(dir_path + '\curated-solubility-dataset.csv')
med = np.median(data['Solubility'])
# If hydrophobic, then label 1, else label 0
data['Solubility'] = np.where(data['Solubility']<=med, 0, 1)
# If hydrophobic, then label 1, else label0

#%%

# Some general drawing parameters. The idea is to draw the molecules
# without any atomic labels and just focusing on partial charge density
# and the color of the bonds (because the atoms are colored). I would expect
# that molecules with high partial charge densities are soluble in water and 
# low partial charge densities are not soluble in water.

n_mols = data.shape[0]
Draw.DrawingOptions.bondLineWidth = 2
Draw.DrawingOptions.atomLabelFontSize = 1
Draw.DrawingOptions.atomLabelMinFontSize = 1
trim_margin = 20

#%%
def shapeParameters():
    """
    This function calculates the appropriate shape of im_tensor so that we
    can appropriately concat all image matrices to it. We need to be able to 
    preassign the shape correctly
    """

    # margin and other parameters. I first generate and image and get the
    # parameters from there. The only user parameter required is the trim
    # width.
    i = 10
    smiles = data['SMILES'][i]
    category = data['Solubility'][i]

    mol = Chem.MolFromSmiles(smiles)
    Chem.ComputeGasteigerCharges(mol)
    contribs = ([float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) 
                for i in range(mol.GetNumAtoms())])
    filename = dir_path + '/figs/mol' + str(i) + '.png'
    fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs,
                                                    scale=150,
                                                    colorMap='bwr', 
                                                    contourLines=1,
                                                    size=(250, 250))
    fig.savefig(filename, bbox_inches='tight')
    
    im = imageio.imread(filename)
    height, width, channels = im.shape
    trimmed_height, trimmed_width = (height - 2*trim_margin, 
                                             width - 2*trim_margin)
    
    return trimmed_width, trimmed_height




# Note that instead of saving and then reloading, we could have just done it
# by directly converting the image into the matrix. But there is something about 
# rdkit not fitting the figure inside the canvas.

# I will have to do the figure saving and the generation together because
# some images are being skipped and then the categories wil not match properly
#%%
def molsToImgsToTensor():
    """
    This function takes all the molecules in the dataset, generates images of
    these molecules and daves these as png images
    """
    # We need to preallocate the tensor in memory because append/concat is very
    # slow. We will have a bunch of elements at the end which we will delete
    # before returning the result by maintaining a counter.
    # %matplotlib agg
    trimmed_height, trimmed_width = shapeParameters()
    n_mols = data.shape[0]  
    im_tensor = torch.zeros((n_mols, trimmed_height, trimmed_width, 4),
                            dtype=torch.uint8)
    category_tensor = torch.zeros((n_mols, 1))
    counter = 0
    for i in tqdm(range(data.shape[0])):
        try:
            smiles = data['SMILES'][i]
            category = torch.Tensor([data['Solubility'][i]]).view(1,-1)
            mol = Chem.MolFromSmiles(smiles)
            Chem.ComputeGasteigerCharges(mol)
            contribs = ([float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge')) 
                        for i in range(mol.GetNumAtoms())])
            filename = dir_path + '/figs/mol' + str(i) + '.png'
            fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs,
                                                            scale=150,
                                                            colorMap='bwr', 
                                                            contourLines=1,
                                                            size=(250, 250))
            fig.savefig(filename, bbox_inches='tight')
            
            im = imageio.imread(filename)
            height, width, channels = im.shape
            
            im = im[trim_margin:-trim_margin, trim_margin:-trim_margin, :]
            im = torch.from_numpy(im).view(1, trimmed_width, trimmed_height, 4)
            im_tensor[counter] = im
            # im_tensor = torch.cat((im_tensor, im), dim=0)
            category_tensor[counter] = category
            # category_tensor = torch.cat((category_tensor, category),
                                        # dim=0)
            counter += 1
        except:
            pass
    return (counter, im_tensor.numpy()[:counter], 
                     category_tensor.int().numpy()[:counter])

#%%
counter, im_tensor, category_tensor = molsToImgsToTensor()

np.save(dir_path + '/im_tensor', im_tensor)
np.save(dir_path + '/category_tensor', category_tensor)

# I still need to remove the frame while converting the figures into 
# torch tensors. For this, I can just import the figures and then trim the edges
# by taking image slices. I will have to go in and delete images that look weird

# Note that there are tons of while pixels, but for now I am just going to
# have to do the convolution with it. Maybe this is also a good chance  to 
# see how well the neural network is able to pick out the features when there
# is such a large set of white pixels. I also noticed that there are some
# images that are fully black. I am also going to leave these in to add 'noise'
# to my dataset to see how robust my code is and how good it is at capturing
# stuff.
#%% [markdown]

# At this point we have a tensor that contains all the images. We then take this
# tensor and convert this into a data training pair in the main file

#%%
