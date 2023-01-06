import os
import re

import networkx as nx
import py3Dmol as pm
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.rdmolops import AddHs, GetAdjacencyMatrix

from functions import _get_rotation_mask, dihedral, rotate_dihedral, write_xyz

fragment_dict = {
    # i
    'TMGA' : 'O=C([2*])[C@H](C/N=C(N(C)C)/N(C)C)N[1*]',
    'DMPGA' : 'O=C([C@@H](N([1*]))C/N=C1N(CCCN/1C)C)[2*]',
    'TMIGA' : 'O=C([C@@H](N([1*]))C/N=C1N(C2=C(C=CC=C2)N/1C)C)[2*]',
    'TPPA' : 'O=C([C@@H](N[1*])CN=P(C1=CC=CC=C1)(C2=CC=CC=C2)C3=CC=CC=C3)[2*]',


    'Azc' : '[O]N1[C@](C2)(C([2*])=O)C[C@@H]3C[C@@]1(C)C[C@H]2C3',
    'PMH' : 'O=C([2*])[C@H](CC1=CN(C)C=N1)N[1*]',
    'DMAA' : 'O=C([2*])[C@H](CN(C)C)N[1*]',
    'Cys' : 'O=C([2*])[C@H](CS)N[1*]',

    # i+1
    'Pro' : 'O=C([2*])[C@H]1N([1*])CCC1',
    'DPro' : 'O=C([2*])[C@@H]1N([1*])CCC1',
    'Pip' : 'O=C([2*])[C@@H]1CCCCN1[1*]',
    'DPip' : 'O=C([2*])[C@H]1CCCCN1[1*]',
    '(4RAcNH)Pro' : 'O=C([2*])[C@@H]1C[C@@H](NC(C)=O)CN1[1*]',
    '(4RAcNH)DPro' : 'O=C([2*])[C@H]1C[C@@H](NC(C)=O)CN1[1*]',
    '(4SAcNH)Pro' : 'O=C([2*])[C@@H]1C[C@H](NC(C)=O)CN1[1*]',
    '(4SAcNH)DPro' : 'O=C([2*])[C@H]1C[C@H](NC(C)=O)CN1[1*]',
    '(4SDFANH)DPro' : 'O=C([C@H]1C[C@H](NC(C(F)F)=O)CN1[1*])[2*]',
    '(4STFANH)DPro' : 'O=C([C@H]1C[C@H](NC(C(F)(F)F)=O)CN1[1*])[2*]',
    '(4SMsNH)DPro' : 'O=C([C@H]1C[C@H](NS(C)(=O)=O)CN1[1*])[2*]',
    '(alphaMe)DPro' : 'O=C([2*])[C@]1(C)N([1*])CCC1',
    'Tic' : 'O=C([C@@H]1CC(C=CC=C2)=C2CN1([1*]))[2*]',
    'Ind' : 'O=C([C@@H](N1[1*])CC2=C1C=CC=C2)[2*]',
    'DInd' : 'O=C([C@H](N1[1*])CC2=C1C=CC=C2)[2*]',

    # i+2/i+3
    'Aic' : 'C([2*])(C1(N[1*])CC2=C(C1)C=CC=C2)=O',
    'Cle' : 'C([2*])(C1(N[1*])CCCC1)=O',
    'Ala' : 'O=C([2*])[C@H](C)N[1*]',
    'DAla' : 'O=C([2*])[C@@H](C)N[1*]',
    'Achc' : 'C([2*])(C1(N[1*])CCCCC1)=O',
    'Acbc' : 'C([2*])(C1(N[1*])CCC1)=O',
    'Aib' : 'O=C([2*])C(C)(C)N([1*])',
    'Acpc' : 'N([1*])C1(C([2*])=O)CC1',
    'Phe' : 'O=C([2*])[C@H](CC1=CC=CC=C1)N[1*]',
    'Tyr(OMe)' : 'O=C([2*])[C@H](CC1=CC=C(OC)C=C1)N[1*]',
    'Tyr(OBn)' : 'O=C([2*])[C@H](CC1=CC=C(OCC2=CC=CC=C2)C=C1)N[1*]',
    'Leu' : 'O=C([2*])[C@H](CC(C)C)N[1*]',
    'DLeu' : 'O=C([2*])[C@@H](CC(C)C)N[1*]',
    'DPhe' : 'O=C([2*])[C@@H](CC1=CC=CC=C1)N[1*]',
    'Phg' : 'O=C([2*])[C@H](C1=CC=CC=C1)N[1*]',
    'DPhg' : 'O=C([2*])[C@@H](C1=CC=CC=C1)N[1*]',
    'Bip' : 'O=C([2*])[C@H](CC1=CC=C(C2=CC=CC=C2)C=C1)N[1*]',
    '1Nal' : 'O=C([2*])[C@H](CC1=C(C=CC=C2)C2=CC=C1)N[1*]',
    '2Nal' : 'O=C([2*])[C@H](CC1=CC(C=CC=C2)=C2C=C1)N[1*]',
    'Tle' : 'O=C([2*])[C@H](C(C)(C)C)N[1*]',
    'DTle' : 'O=C([2*])[C@@H](C(C)(C)C)N[1*]',
    'Val' : 'O=C([C@@H](N([1*]))C(C)C)([2*])',
    'DVal' : 'O=C([C@H](N([1*]))C(C)C)([2*])',

    # C-caps
    'OH' : 'O([1*])',
    'OMe' : 'CO([1*])',
    'NMe2' : 'N(C)(C)([1*])',
    'NHMe' : 'N(C)([1*])',
    'NHPh' : 'N([1*])C1=CC=CC=C1',
    'NHiPr' : 'CC(C)N[1*]',
    'NHtBu' : 'CC(C)(C)N[1*]',
    'NEt2' : 'CCN(CC)[1*]',
    'N(CH2)4' : 'N([1*])1CCCC1',
    'N(CH2)5' : 'N([1*])1CCCCC1',
    'NHCH2CF3': 'N([1*])CC(F)(F)F',
    'NHCH2CF2CF3' : 'N([1*])CC(F)(F)C(F)(F)F',
    '(R)PEA' : 'C[C@H](C1=CC=CC=C1)N[1*]',
    '(S)PEA' : 'C[C@@H](C1=CC=CC=C1)N[1*]',
    'NHBn' : 'N([1*])CC1=CC=CC=C1',

    # N-caps
    'Boc' : 'C(C)(C)(C)OC(=O)([2*])',
    'Fmoc' : 'O=C([2*])OCC1C2=CC=CC=C2C3=CC=CC=C13',
    'PhUrea' : 'C([2*])(NC1=CC=CC=C1)=O',
    'tBuUrea' : 'C([2*])(NC(C)(C)C)=O',
    'PhOCO' : 'C([2*])(OC1=CC=CC=C1)=O',
    'Ts' : 'O=S(C1=CC=C(C)C=C1)([2*])=O',
    'Ms' : 'CS(=O)([2*])=O',
}

frags_mols = [Chem.MolFromSmiles(smi) for smi in fragment_dict.values()]

turn_angles_dict = {

    'one prime beta hairpin' : ((
                                45, # H-N(term)-C(α,i)-C(O)
                                147, # N(term)-C(a,i)-C(O)-N(i+1)
                                171, # C(a,i)-C(O)-N(i+1)-C(a,i+1)
                                -167, # N(i+1)-C(a,i+1)-C(O)-O
                                -175, # C(a,i+1)-C(O)-N(i+2)-C(a,i+2)
                                74, # C(O)-N(i+2)-C(a,i+2)-C(O)
                                33, # N(i+2)-C(a,i+2)-C(O)-N(i+3)
                                -156, # C(O)-N(i+3)-C(a,i+3)-C(O)
                                -9, # N(i+3)-C(a,i+3)-C(O)-O
                            ),),

    'two prime beta hairpin' : ((
                                70, # H-N(term)-C(α,i)-C(O)
                                111, # N(term)-C(a,i)-C(O)-N(i+1)
                                -177, # C(a,i)-C(O)-N(i+1)-C(a,i+1)
                                100, # N(i+1)-C(a,i+1)-C(O)-O
                                170, # C(a,i+1)-C(O)-N(i+2)-C(a,i+2)
                                -165, # C(O)-N(i+2)-C(a,i+2)-C(O)
                                33, # N(i+2)-C(a,i+2)-C(O)-N(i+3)
                                -113, # C(O)-N(i+3)-C(a,i+3)-C(O)
                                -40, # N(i+3)-C(a,i+3)-C(O)-O
                            ),(
                                57,
                                136,
                                -167,
                                99,
                                174,
                                -176,
                                47,
                                -135,
                                -26,
                            ))


}

def build_peptide(string):

    smi_list = [fragment_dict[frag] for frag in reversed(string.split('-'))]
    return smiles_joiner(smi_list)

def correct_amides(coords, graph, rdkit_mol):
    '''
    Rotate all secondary amides (and carbamates) in the more stable trans conformation.
    '''
    amides = rdkit_mol.GetSubstructMatches(Chem.MolFromSmarts('N([H])C(=O)'))
    amides = [[b, a, c, d] for a, b, c, d in amides]
    for quadruplet in amides:
        current_angle = dihedral(coords[quadruplet])
        mask = _get_rotation_mask(graph, quadruplet)
        coords = rotate_dihedral(coords, quadruplet, 180-current_angle, mask=mask)
    return coords

def smiles_joiner(list_of_fragments, n=9):
    # pick the first two fragments to join and replace [n*] with n

    # print(f'lof is {list_of_fragments}')
    f1 = re.sub('\[1\*\]', str(n), list_of_fragments[0])
    f2 = re.sub('\[2\*\]', str(n), list_of_fragments[1])

    # remove parentheses around 9 if present
    f1 = re.sub(f'\({n}\)', str(n), f1)
    f2 = re.sub(f'\({n}\)', str(n), f2)
    out = '.'.join([f1, f2])
    # print(f'joined out is {out} ')

    # if the SMILES string starts with a 9, move it after the atom label
    out = re.sub('(^9)(\w)', r'\g<2>9', out)
    # print(f'refined out is {out}\n')

    # Go through RDKit to obtain a better, canonical SMILES for the new molecule
    tmp = Chem.MolFromSmiles(out)
    out = Chem.MolToSmiles(tmp)
    # print(f'rdkit out is {out} ')

    # iterate the process if we have more fragments to join
    if len(list_of_fragments) == 2:
        return out
    
    return smiles_joiner([out, *list_of_fragments[2:]])

def draw(pep):
    return Chem.Draw.MolToImage(Chem.MolFromSmiles(pep), size=(500, 300))

def get_beta_turn_torsion_indices(rdkit_mol):
    '''
    '''
    pro_pattern = Chem.MolFromSmarts('N([H])CC(=O)N1[#6][#6]CC1C(=O)NCC(=O)N([H])CC(=O)')
    match = rdkit_mol.GetSubstructMatch(pro_pattern)

    quadruplets = [(1, 0, 2, 3), # H-N(term)-C(α,i)-C(O)
                    (0, 2, 3, 5), # N(term)-C(α,i)-C(O)-N(i+1)
                    (2, 3, 5, 9), # C(a,i)-C(O)-N(i+1)-C(a,i+1)
                    (5, 9, 10, 11), # N(i+1)-C(a,i+1)-C(O)-O
                    (9, 10, 12, 13), # C(a,i+1)-C(O)-N(i+2)-C(a,i+2)
                    (10, 12, 13, 14), # C(O)-N(i+2)-C(a,i+2)-C(O)
                    (12, 13, 14, 16), # N(i+2)-C(a,i+2)-C(O)-N(i+3)
                    (14, 16, 18, 19), # C(O)-N(i+3)-C(a,i+3)-C(O)
                    (16, 18, 19, 20)] # N(i+3)-C(a,i+3)-C(O)-O

    if not match:
        pip_pattern = Chem.MolFromSmarts('N([H])CC(=O)N1[#6][#6]CCC1C(=O)NCC(=O)N([H])CC(=O)')
        match = rdkit_mol.GetSubstructMatch(pip_pattern)

        quadruplets = [(1, 0, 2, 3), # H-N(term)-C(α,i)-C(O)
                        (0, 2, 3, 5), # N(term)-C(α,i)-C(O)-N(i+1)
                        (2, 3, 5, 10), # C(a,i)-C(O)-N(i+1)-C(a,i+1)
                        (5, 10, 11, 12), # N(i+1)-C(a,i+1)-C(O)-O
                        (10, 11, 13, 14), # C(a,i+1)-C(O)-N(i+2)-C(a,i+2)
                        (11, 13, 14, 15), # C(O)-N(i+2)-C(a,i+2)-C(O)
                        (13, 14, 15, 17), # N(i+2)-C(a,i+2)-C(O)-N(i+3)
                        (15, 17, 19, 20), # C(O)-N(i+3)-C(a,i+3)-C(O)
                        (17, 19, 20, 21)] # N(i+3)-C(a,i+3)-C(O)-O

    if not match:
        return Exception()


    return [[match[i] for i in angles] for angles in quadruplets]

def get_beta_hairpin_hb(rdkit_mol):
    '''
    '''
    pro_pattern = Chem.MolFromSmarts('N([H])CC(=O)N1[#6][#6]CC1C(=O)NCC(=O)N([H])CC(=O)')
    end = 20
    match = rdkit_mol.GetSubstructMatch(pro_pattern)

    if not match:
        pip_pattern = Chem.MolFromSmarts('N([H])CC(=O)N1[#6][#6]CCC1C(=O)NCC(=O)N([H])CC(=O)')
        end = 21
        match = rdkit_mol.GetSubstructMatch(pip_pattern)

    if not match:
        return Exception()

    return (match[1], match[end])

def smiles_graphize(smiles, atomnos):
    '''
    '''

    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = AddHs(rdkit_mol)
    matrix = GetAdjacencyMatrix(rdkit_mol)
    graph = nx.from_numpy_matrix(matrix)
    nx.set_node_attributes(graph, dict(enumerate(atomnos)), 'atomnos')

    return graph

def measure_turn_angles(coords, atomnos, turn_pattern, angle_ids):
    '''
    '''
    pattern_mol = Chem.MolFromSmarts(turn_pattern)
    with open('temp.xyz', 'w') as f:
        write_xyz(coords, atomnos, f)

    os.system('obabel temp.xyz -O temp.mol')

    rdkit_mol = Chem.AddHs(Chem.MolFromMolFile('temp.mol'))
    
    match = rdkit_mol.GetSubstructMatch(pattern_mol)
  
    if not match:
        return Exception()


    return [round(dihedral(coords[[match[i] for i in angles]])) for angles in angle_ids]

def show(filename):
    # pm.view(data=open('tmp.xyz').readlines())
    with open(filename) as f:
        data = f.read()
    view = pm.view(data=data, style='Ball')
    view.setStyle({'stick':{}})
    return view