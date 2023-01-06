from copy import deepcopy

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from functions import (_get_rotation_mask, _get_torsions, clean_directory,
                       compenetration_check, dihedral, norm_of, openbabel_opt,
                       read_xyz, rotate_dihedral, sanitize_conformer,
                       write_xyz)
from lib import (build_peptide, get_beta_hairpin_hb,
                      get_beta_turn_torsion_indices, smiles_graphize,
                      turn_angles_dict, correct_amides)


def run(smiles:str, name='', turn='two prime beta hairpin', tweak=True, opt=False):

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

    print('Rotating peptide in the right conformation...')

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

    print(f'Sanitizing other rotable bonds...')

    # sanitize it
    coords = sanitize_conformer(coords, torsions, graph)

    if tweak:

        print(f'Tweaking dihedrals...')

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
        print(f'Performing a geometry optimization...')
        # optimize it
        coords = openbabel_opt(coords, data.atomnos, constrained_indexes=[hairpin_hb], constrained_distances=[2.0])
        coords = openbabel_opt(coords, data.atomnos, [])
        d = abs(norm_of(coords[hairpin_hb[0]] - coords[hairpin_hb[1]]) - 2)
        print(f'OPT -> Optimized Δd to {round(d,2)} A')

    with open(f'{name}.xyz', 'w') as f:
        write_xyz(coords, data.atomnos, f, title='Embedded via SAAD')

    clean_directory()

if __name__ == '__main__':

    peptide = input('Sequence: ')
    smiles = build_peptide(peptide)
    run(smiles, turn='two prime beta hairpin', tweak=True, opt=True)