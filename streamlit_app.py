from copy import deepcopy

import numpy as np
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole

from functions import (_get_rotation_mask, _get_torsions, clean_directory,
                       compenetration_check, dihedral, norm_of, openbabel_opt,
                       read_xyz, rotate_dihedral, sanitize_conformer,
                       write_xyz)
from lib import (build_peptide, correct_amides, draw, fragment_dict,
                 get_beta_hairpin_hb, get_beta_turn_torsion_indices, show,
                 smiles_graphize, turn_angles_dict)


@st.cache
def get_frags_image():
    return Draw.MolsToGridImage(
                            frags_mols,
                            legends=list(fragment_dict.keys()),
                            molsPerRow=3,
                            subImgSize=(400,400),
                            maxMols=200,
                            returnPNG=False,
                            )

@st.cache
def streamlit_run(smiles, name, turn, tweak, opt):

    with st.spinner(text="Generating peptide..."):

        mw = Chem.RWMol(Chem.MolFromSmiles(smiles))
        mw = Chem.AddHs(mw)
        AllChem.EmbedMolecule(mw)

        indices_to_turn = get_beta_turn_torsion_indices(mw)
        hairpin_hb = get_beta_hairpin_hb(mw)

        # Sometimes the first embedding fails
        if len(mw.GetConformers()) == 0:
            AllChem.EmbedMolecule(mw)

        with open('temp.xyz', 'w') as f:
            f.write(Chem.MolToXYZBlock(mw))

        data = read_xyz('temp.xyz')
        coords = data.atomcoords[0]

        graph = smiles_graphize(smiles, data.atomnos)

    with st.spinner(text="Rotating peptide in the right conformation..."):

        # rotate amides the right way
        coords = correct_amides(coords, graph, mw)

        min_d = abs(norm_of(coords[hairpin_hb[0]] - coords[hairpin_hb[1]]) - 2)
        for angle_set in turn_angles_dict[turn]:
            temp_coords = deepcopy(coords)
            for quadruplet, angle in zip(indices_to_turn, angle_set):
                current_angle = dihedral(temp_coords[quadruplet])
                mask = _get_rotation_mask(graph, quadruplet)
                rotation = angle-current_angle
                temp_coords = rotate_dihedral(temp_coords, quadruplet, rotation, mask=mask)

            d = abs(norm_of(temp_coords[hairpin_hb[0]] - temp_coords[hairpin_hb[1]]) - 2)
            if d < min_d:
                min_d = d
                coords = temp_coords
                print(f'EMBED -> Embedded with Δd = {round(d,2)} A')

        graph.add_edge(*hairpin_hb)
        torsions = _get_torsions(graph, [], [])

    with st.spinner(text="Sanitizing other rotable bonds..."):

        # sanitize it to remove clashes/compenetrations
        coords = sanitize_conformer(coords, torsions, graph)

    if tweak:

        with st.spinner(text="Tweaking dihedrals..."):

            # restore initial graph with no hbs added
            graph.remove_edge(*hairpin_hb)

            # tweak angles to get best starting guess, that is hairpin HB close to 2 A
            min_d = abs(norm_of(coords[hairpin_hb[0]] - coords[hairpin_hb[1]]) - 2)
            delta = 10
            for iteration in range(1000):
                if iteration == 500 and min_d < 2:
                    delta = 5
                elif iteration == 100 and min_d < 1:
                    delta = 1

                temp_coords = deepcopy(coords)
                
                perturbation = np.random.rand(len(turn_angles_dict[turn][0]))*(2*delta)-delta
                new_angles = np.array(turn_angles_dict[turn][0]) + perturbation

                for quadruplet, angle in zip(indices_to_turn, new_angles):
                    current_angle = dihedral(temp_coords[quadruplet])
                    mask = _get_rotation_mask(graph, quadruplet)
                    rotation = angle-current_angle
                    temp_coords = rotate_dihedral(temp_coords, quadruplet, rotation, mask=mask)

                current_angles = [dihedral(temp_coords[quadruplet]) for quadruplet in indices_to_turn]

                d = abs(norm_of(temp_coords[hairpin_hb[0]] - temp_coords[hairpin_hb[1]]) - 2)
                
                if d < min_d and compenetration_check(temp_coords):
                    min_d = d
                    coords = temp_coords
                    print(f'TWEAK -> Improved Δd to {round(d,2)} A')

    if opt:
        with st.spinner(text="Performing a FF geometry optimization..."):
            # optimize it
            coords = openbabel_opt(coords, data.atomnos, constrained_indexes=[hairpin_hb], constrained_distances=[2.0])
            coords = openbabel_opt(coords, data.atomnos, [])
            d = abs(norm_of(coords[hairpin_hb[0]] - coords[hairpin_hb[1]]) - 2)
            print(f'OPT -> Optimized Δd to {round(d,2)} A')

    with open(f'{name}.xyz', 'w') as f:
        write_xyz(coords, data.atomnos, f, title='Embedded via SAAD')

    clean_directory()

    return f'{name}.xyz'

def post_run(filename, smiles):

    # from stmol import showmol
    # showmol(show(filename), height=400, width=400)

    st.text('3D widget coming soon...')

    with open(filename, "rb") as file:
        st.download_button(
                label='Download 3D structure (.xyz)',
                data=file,
                file_name=filename,
            )
    
    run_button = not st.button('Reset')

if __name__ == "__main__":

    st.title('BeTTeR - *Be*ta *T*urn *Te*tramers *R*otator')
    st.write('### Generate pre-folded beta-turn peptide conformations')
    st.write('© Nicolò Tampellini - 2023')

    tab1, tab2 = st.tabs([':gear: Embedder', ':pill: Amino acids and C/N caps'])

    with tab1:

        options = ['Amino acid sequence', 'SMILES']
        input_format = st.selectbox('Input format', options)

        defaults = {
            'Amino acid sequence':'Boc-TMGA-(alphaMe)DPro-Acpc-Phe-NMe2',
            'SMILES':'CNC(=O)[C@H](Cc1ccccc1)NC(=O)C(C)(C)NC(=O)[C@H]1Cc2ccccc2N1C(=O)[C@H](CN=C1N(C)CCCN1C)NC(=O)Nc1ccccc1',
            }
        input_string = str(st.text_input('Input',value=defaults[input_format]))

        turn_dict = {
            'Type I\' β-Hairpin' : 'one prime beta hairpin',
            'Type II\' β-Hairpin' : 'two prime beta hairpin',
            }
        turn = st.selectbox('Turn type', turn_dict.keys())

        tweak = st.checkbox('Autocorrect dihedrals after embedding (a bit slower)', value=True)
        opt = st.checkbox('Force Field optimization with Openbabel (slower but better)', value=False)

        if st.button('Run'):
            if input_format == 'Amino acid sequence':
                name = input_string
                smiles = build_peptide(input_string)
            else:
                name = 'SMILES_peptide'
                smiles = input_string
        
            st.image(Chem.Draw.MolToImage(Chem.MolFromSmiles(smiles), size=(500, 500)))

            filename = streamlit_run(smiles, name=name, turn=turn_dict[turn], tweak=tweak, opt=opt)
            # filename = 'Boc-TMGA-(alphaMe)DPro-Acpc-Phe-NMe2.xyz'

            post_run(filename, smiles)

    with tab2:

        frags_mols = [Chem.MolFromSmiles(smi) for smi in fragment_dict.values()]
        im = get_frags_image() # so this is cached
        st.image(im, use_column_width=True)