import os
from copy import deepcopy
from itertools import combinations
from math import sqrt

import networkx as nx
import numba as nb
import numpy as np
from cclib.io import ccread
# from openbabel import openbabel as ob
from periodictable import core, covalent_radius, mass

pt = core.PeriodicTable(table="H=1")
covalent_radius.init(pt)
mass.init(pt)

def write_xyz(coords:np.array, atomnos:np.array, output, title='temp'):
    '''
    output is of _io.TextIOWrapper type

    '''
    assert atomnos.shape[0] == coords.shape[0]
    assert coords.shape[1] == 3
    string = ''
    string += str(len(coords))
    string += f'\n{title}\n'
    for i, atom in enumerate(coords):
        string += '%s     % .6f % .6f % .6f\n' % (pt[atomnos[i]].symbol, atom[0], atom[1], atom[2])
    output.write(string)

def read_xyz(filename):
    mol = ccread(filename)
    assert mol is not None, f'Reading molecule {filename} failed - check its integrity.'
    return mol

def rotate_dihedral(coords, dihedral, angle, mask=None, indexes_to_be_moved=None):
    '''
    Rotate a molecule around a given bond.
    Atoms that will move are the ones
    specified by mask or indexes_to_be_moved.
    If both are None, only the first index of
    the dihedral iterable is moved.

    angle: angle, in degrees
    '''

    i1, i2, i3 ,_ = dihedral

    if indexes_to_be_moved is not None:
        mask = np.array([i in indexes_to_be_moved for i, _ in enumerate(coords)])

    if mask is None:
        mask = i1

    axis = coords[i2] - coords[i3]
    mat = rot_mat_from_pointer(axis, angle)
    # mat = rot_mat_from_pointer_scipy(axis, angle)
    center = coords[i3]

    coords[mask] = (mat @ (coords[mask] - center).T).T + center

    return coords

def _get_rotation_mask(graph, torsion):
    '''
    '''
    i1, i2, i3, _ = torsion

    graph.remove_edge(i2, i3)
    reachable_indexes = nx.shortest_path(graph, i1).keys()
    # get all indexes reachable from i1 not going through i2-i3

    graph.add_edge(i2, i3)
    # restore modified graph

    mask = np.array([i in reachable_indexes for i in graph.nodes], dtype=bool)
    # generate boolean mask

    # if np.count_nonzero(mask) > int(len(mask)/2):
    #     mask = ~mask
    # if we want to rotate more than half of the indexes,
    # invert the selection so that we do less math

    mask[i2] = False
    # do not rotate i2: would not move,
    # since it lies on rotation axis
    
    return mask

@nb.njit
def vec_angle(v1, v2):
    v1_u = norm(v1)
    v2_u = norm(v2)
    return np.arccos(clip(np.dot(v1_u, v2_u), -1.0, 1.0))*180/np.pi

@nb.njit
def clip(n, lower, higher):
    '''
    jittable version of np.clip for single values
    '''
    if n > higher:
        return higher
    elif n < lower:
        return lower
    else:
        return n

@nb.njit(fastmath=True)
def all_dists(A, B):
    assert A.shape[1]==B.shape[1]
    C=np.empty((A.shape[0],B.shape[0]),A.dtype)
    I_BLK=32
    J_BLK=32
    
    #workaround to get the right datatype for acc
    init_val_arr=np.zeros(1,A.dtype)
    init_val=init_val_arr[0]
    
    #Blocking and partial unrolling
    #Beneficial if the second dimension is large -> computationally bound problem 
    # 
    for ii in nb.prange(A.shape[0]//I_BLK):
        for jj in range(B.shape[0]//J_BLK):
            for i in range(I_BLK//4):
                for j in range(J_BLK//2):
                    acc_0=init_val
                    acc_1=init_val
                    acc_2=init_val
                    acc_3=init_val
                    acc_4=init_val
                    acc_5=init_val
                    acc_6=init_val
                    acc_7=init_val
                    for k in range(A.shape[1]):
                        acc_0+=(A[ii*I_BLK+i*4+0,k] - B[jj*J_BLK+j*2+0,k])**2
                        acc_1+=(A[ii*I_BLK+i*4+0,k] - B[jj*J_BLK+j*2+1,k])**2
                        acc_2+=(A[ii*I_BLK+i*4+1,k] - B[jj*J_BLK+j*2+0,k])**2
                        acc_3+=(A[ii*I_BLK+i*4+1,k] - B[jj*J_BLK+j*2+1,k])**2
                        acc_4+=(A[ii*I_BLK+i*4+2,k] - B[jj*J_BLK+j*2+0,k])**2
                        acc_5+=(A[ii*I_BLK+i*4+2,k] - B[jj*J_BLK+j*2+1,k])**2
                        acc_6+=(A[ii*I_BLK+i*4+3,k] - B[jj*J_BLK+j*2+0,k])**2
                        acc_7+=(A[ii*I_BLK+i*4+3,k] - B[jj*J_BLK+j*2+1,k])**2
                    C[ii*I_BLK+i*4+0,jj*J_BLK+j*2+0]=np.sqrt(acc_0)
                    C[ii*I_BLK+i*4+0,jj*J_BLK+j*2+1]=np.sqrt(acc_1)
                    C[ii*I_BLK+i*4+1,jj*J_BLK+j*2+0]=np.sqrt(acc_2)
                    C[ii*I_BLK+i*4+1,jj*J_BLK+j*2+1]=np.sqrt(acc_3)
                    C[ii*I_BLK+i*4+2,jj*J_BLK+j*2+0]=np.sqrt(acc_4)
                    C[ii*I_BLK+i*4+2,jj*J_BLK+j*2+1]=np.sqrt(acc_5)
                    C[ii*I_BLK+i*4+3,jj*J_BLK+j*2+0]=np.sqrt(acc_6)
                    C[ii*I_BLK+i*4+3,jj*J_BLK+j*2+1]=np.sqrt(acc_7)
        #Remainder j
        for i in range(I_BLK):
            for j in range((B.shape[0]//J_BLK)*J_BLK,B.shape[0]):
                acc_0=init_val
                for k in range(A.shape[1]):
                    acc_0+=(A[ii*I_BLK+i,k] - B[j,k])**2
                C[ii*I_BLK+i,j]=np.sqrt(acc_0)
    
    #Remainder i
    for i in range((A.shape[0]//I_BLK)*I_BLK,A.shape[0]):
        for j in range(B.shape[0]):
            acc_0=init_val
            for k in range(A.shape[1]):
                acc_0+=(A[i,k] - B[j,k])**2
            C[i,j]=np.sqrt(acc_0)
            
    return C

@nb.njit
def compenetration_check(coords, max_clashes=0):
    return 0 if np.count_nonzero(
                                 (all_dists(coords,coords) < 1) & (
                                  all_dists(coords,coords) > 0)
                                ) > max_clashes else 1

def scramble_check(coords, atomnos, graph, max_newbonds=0, hydrogen_bonds=None) -> bool:
    '''
    Check if a multimolecular arrangement has scrambled. If more 
    than a given number of bonds changed (formed or broke) the
    structure is considered scrambled, and the method returns False.
    '''

    new_bonds = graphize(coords, atomnos)
    delta_bonds = (graph.edges | new_bonds) - (graph.edges & new_bonds)

    hydrogen_bonds = hydrogen_bonds or []
    for bond in delta_bonds.copy():
        for a1, a2 in hydrogen_bonds:
            if (a1 in bond) or (a2 in bond):
                delta_bonds -= {bond}
    # removing bonds involving constrained atoms: they are not counted as scrambled bonds

    if len(delta_bonds) > max_newbonds:
        return False
    return True

@nb.njit
def dihedral(p):
    '''
    Returns dihedral angle in degrees from 4 3D vecs
    Praxeolitic formula: 1 sqrt, 1 cross product
    
    '''
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    # normalize b1 so that it does not influence magnitude of vector
    # rejections that come next
    b1 /= norm_of(b1)

    # vector rejections
    # v = projection of b0 onto plane perpendicular to b1
    #   = b0 minus component that aligns with b1
    # w = projection of b2 onto plane perpendicular to b1
    #   = b2 minus component that aligns with b1
    v = b0 - np.dot(b0, b1)*b1
    w = b2 - np.dot(b2, b1)*b1

    # angle between v and w in a plane is the torsion angle
    # v and w may not be normalized but that's fine since tan is y/x
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    
    return np.degrees(np.arctan2(y, x))

@nb.njit
def norm(vec):
    '''
    Returns the normalized vector.
    Reasonably faster than Numpy version.
    Only for 3D vectors.
    '''
    return vec / sqrt((vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]))

@nb.njit
def norm_of(vec):
    '''
    Returns the norm of the vector.
    Reasonably faster than Numpy version.
    Only for 3D vectors.
    '''
    return sqrt((vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]))

@nb.njit
def rot_mat_from_pointer(pointer, angle):
    '''
    Returns the rotation matrix that rotates a system around the given pointer
    of angle degrees. The algorithm is based on scipy quaternions.
    :params pointer: a 3D vector
    :params angle: an int/float, in degrees
    :return rotation_matrix: matrix that applied to a point, rotates it along the pointer
    '''
    assert pointer.shape[0] == 3

    pointer = norm(pointer)
    angle *= np.pi/180
    quat = np.array([np.sin(angle/2)*pointer[0],
                     np.sin(angle/2)*pointer[1],
                     np.sin(angle/2)*pointer[2],
                     np.cos(angle/2)])
    # normalized quaternion, scalar last (i j k w)
    
    return quaternion_to_rotation_matrix(quat)

@nb.njit
def quaternion_to_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q (adjusting for scalar last in input)
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return np.ascontiguousarray(rot_matrix)

def d_min_bond(e1, e2):
    return 1.2 * (pt[e1].covalent_radius + pt[e2].covalent_radius)

class Torsion:
    def __repr__(self):
        if hasattr(self, 'n_fold'):
            return f'Torsion({self.i1}, {self.i2}, {self.i3}, {self.i4}; {self.n_fold}-fold)'
        return f'Torsion({self.i1}, {self.i2}, {self.i3}, {self.i4})'

    def __init__(self, i1, i2, i3, i4):
        self.i1 = i1
        self.i2 = i2
        self.i3 = i3
        self.i4 = i4
        self.torsion = (i1, i2, i3 ,i4)

    def in_cycle(self, graph):
        '''
        Returns True if the torsion is part of a cycle
        '''
        graph.remove_edge(self.i2, self.i3)
        cyclical = nx.has_path(graph, self.i1, self.i4)
        graph.add_edge(self.i2, self.i3)
        return cyclical

    def is_rotable(self, graph, hydrogen_bonds) -> bool:
        '''
        hydrogen bonds: iterable with pairs of sorted atomic indexes
        '''

        if sorted((self.i2, self.i3)) in hydrogen_bonds:
            # self.n_fold = 6
            # # This has to be an intermolecular HB: rotate it
            # return True
            return False

        if _is_free(self.i2, graph) or (
           _is_free(self.i3, graph)):

            if _is_nondummy(self.i2, self.i3, graph) and (
               _is_nondummy(self.i3, self.i2, graph)):

                self.n_fold = self.get_n_fold(graph)
                return True

        return False

    def get_n_fold(self, graph) -> int:

        nums = (graph.nodes[self.i2]['atomnos'],
                graph.nodes[self.i3]['atomnos'])

        if 1 in nums:
            return 6 # H-N, H-O hydrogen bonds
        
        if is_amide_n(self.i2, graph, mode=2) or (
           is_amide_n(self.i3, graph, mode=2)):
           # tertiary amides rotations are 2-fold
           return 2

        if (6 in nums) or (7 in nums) or (16 in nums): # if C, N or S atoms

            sp_n_i2 = get_sp_n(self.i2, graph)
            sp_n_i3 = get_sp_n(self.i3, graph)

            if 3 in (sp_n_i2, sp_n_i3): # Csp3-X, Nsp3-X, Ssulfone-X
                return 3

        return 4 #O-O, S-S, Ar-Ar, Ar-CO, and everything else

    def get_angles(self):
        return {
                2:(0, 180),
                3:(0, 120, 240),
                4:(0, 90, 180, 270),
                6:(0, 60, 120, 180, 240, 300),
                }[self.n_fold]

    def sort_torsion(self, graph, constrained_indexes) -> None:
        '''
        Acts on the self.torsion tuple leaving it as it is or
        reversing it, so that the first index of it (from which
        rotation will act) is external to the molecule constrained
        indexes. That is we make sure to rotate external groups
        and not the whole transition state.
        '''
        graph.remove_edge(self.i2, self.i3)
        for d in constrained_indexes.flatten():
            if nx.has_path(graph, self.i2, d):
                self.torsion = tuple(reversed(self.torsion))
        graph.add_edge(self.i2, self.i3)

def _is_free(index, graph):
    '''
    Return True if the index specified
    satisfies all of the following:
    - Is not a sp2 carbonyl carbon atom
    - Is not the oxygen atom of an ester
    - Is not the nitrogen atom of a secondary amide (CONHR)

    '''
    if all((
            graph.nodes[index]['atomnos'] == 6,
            is_sp_n(index, graph, 2),
            8 in (graph.nodes[n]['atomnos'] for n in neighbors(graph, index))
          )):
        return False

    if is_amide_n(index, graph, mode=1):
        return False

    if is_ester_o(index, graph):
        return False

    return True

def _is_nondummy(i, root, graph) -> bool:
    '''
    Checks that a molecular rotation along the dihedral
    angle (*, root, i, *) is non-dummy, that is the atom
    at index i, in the direction opposite to the one leading
    to root, has different substituents. i.e. methyl, CF3 and tBu
    rotations should return False.
    '''

    if graph.nodes[i]['atomnos'] not in (6,7):
        return True
    # for now, we only discard rotations around carbon
    # and nitrogen atoms, like methyl/tert-butyl/triphenyl
    # and flat symmetrical rings like phenyl, N-pyrrolyl...

    G = deepcopy(graph)
    nb = neighbors(G, i)
    nb.remove(root)

    if len(nb) == 1:
        if len(neighbors(G, nb[0])) == 2:
            return False
    # if node i has two bonds only (one with root and one with a)
    # and the other atom (a) has two bonds only (one with i)
    # the rotation is considered dummy: some other rotation
    # will account for its freedom (i.e. alkynes, hydrogen bonds)

    for n in nb:
        G.remove_edge(i, n)

    subgraphs_nodes = [_set for _set in nx.connected_components(G)
                       if not root in _set]

    if len(subgraphs_nodes) == 1:
        return True
        # if not, the torsion is likely to be rotable
        # (tetramethylguanidyl alanine C(β)-N bond)

    subgraphs = [nx.subgraph(G, s) for s in subgraphs_nodes]
    for sub in subgraphs[1:]:
        if not nx.is_isomorphic(subgraphs[0], sub,
                                node_match=lambda n1, n2: n1['atomnos'] == n2['atomnos']):
            return True
    # Care should be taken because chiral centers are not taken into account: a rotation 
    # involving an index where substituents only differ by stereochemistry, and where a 
    # rotation is not an element of symmetry of the subsystem, the rotation is discarded
    # even if it would be meaningful to keep it.

    return False

def _get_hydrogen_bonds(coords, atomnos, graph, d_min=2.5, d_max=3.3, max_angle=45, fragments=None):
    '''
    Returns a list of tuples with the indexes
    of hydrogen bonding partners.

    An HB is a pair of atoms:
    - with one H and one X (N or O) atom
    - with an Y-X distance between d_min and d_max (i.e. N-O, Angstroms)
    - with an Y-H-X angle below max_angle (i.e. N-H-O, degrees)

    If fragments is specified (iterable of iterable of indexes for each fragment)
    the function only returns inter-fragment hydrogen bonds.
    '''

    hbs = []
    # initializing output list

    het_idx = np.array([i for i, a in enumerate(atomnos) if a in (7,8)], dtype=int)
    # Indexes where N or O atoms are present. Let's ignore F for now.

    for i, i1 in enumerate(het_idx):
        for i2 in het_idx[i+1:]:

            if fragments is not None:
                if any(((i1 in f and i2 in f) for f in fragments)):
                    continue
            # if inter-fragment HBs are requested, skip intra-HBs

            if d_min < norm_of(coords[i1]-coords[i2]) < d_max:
            # getting all pairs of O/N atoms between these distances

                Hs = [i for i in (neighbors(graph, i1) +
                                  neighbors(graph, i2)) if graph.nodes[i]['atomnos'] == 1]
                # getting the indexes of all H atoms attached to them

                versor = norm(coords[i2]-coords[i1])
                # versor connectring the two Heteroatoms

                for iH in Hs:

                    v1 = coords[iH]-coords[i1]
                    v2 = coords[iH]-coords[i2]
                    # vectors connecting heteroatoms to H

                    d1 = norm_of(v1)
                    d2 = norm_of(v2)
                    # lengths of these vectors

                    l1 = v1 @ versor
                    l2 = v2 @ -versor
                    # scalar projection in the heteroatom direction

                    alfa = vec_angle(v1, versor) if l1 < l2 else vec_angle(v2, -versor)
                    # largest planar angle between Het-H and Het-Het, in degrees (0 to 90°)

                    if alfa < max_angle:
                    # if the three atoms are not too far from being in line

                        if d1 < d2:
                            hbs.append(sorted((iH,i2)))
                        else:
                            hbs.append(sorted((iH,i1)))
                        # adding the correct pair of atoms to results

                        break

    return hbs

def _get_quadruplets(graph):
    '''
    Returns list of quadruplets that indicate potential torsions
    '''

    allpaths = []
    for node in graph:
        allpaths.extend(findPaths(graph, node, 3))
    # get all possible continuous indexes quadruplets

    quadruplets, q_ids = [], []
    for path in allpaths:
        _, i2, i3, _ = path
        q_id = tuple(sorted((i2, i3)))

        if (q_id not in q_ids):

            quadruplets.append(path)
            q_ids.append(q_id)

    # Yields non-redundant quadruplets
    # Rejects (4,3,2,1) if (1,2,3,4) is present

    return np.array(quadruplets)

def _get_torsions(graph, hydrogen_bonds, double_bonds):
    '''
    Returns list of Torsion objects
    '''
    
    torsions = []
    for path in _get_quadruplets(graph):
        _, i2, i3, _ = path
        bt = tuple(sorted((i2, i3)))

        if bt not in double_bonds:
            t = Torsion(*path)

            if (not t.in_cycle(graph)) and t.is_rotable(graph, hydrogen_bonds):
                torsions.append(t)
    # Create non-redundant torsion objects
    # Rejects (4,3,2,1) if (1,2,3,4) is present
    # Rejects torsions that do not represent a rotable bond

    return torsions

def neighbors(graph, index):
    # neighbors = list([(a, b) for a, b in graph.adjacency()][index][1].keys())
    neighbors = list(graph.neighbors(index))
    if index in neighbors:
        neighbors.remove(index)
    return neighbors

def is_sp_n(index, graph, n):
    '''
    Returns True if the sp_n value matches the input
    '''
    sp_n = get_sp_n(index, graph)
    if sp_n == n:
        return True
    return False

def get_sp_n(index, graph):
    '''
    Returns n, that is the apex of sp^n hybridization for CONPS atoms.
    This is just an assimilation to the carbon geometry in relation to sp^n:
    - sp(1) is linear
    - sp2 is planar
    - sp3 is tetraedral
    This is mainly used to understand if a torsion is to be rotated or not.
    '''
    element = graph.nodes[index]['atomnos']

    if element not in (6,7,8,15,16):
        return None

    d = {
        6:{2:1, 3:2, 4:3},      # C - 2 neighbors means sp, 3 nb means sp2, 4 nb sp3
        7:{2:2, 3:3, 4:3},      # N - 2 neighbors means sp2, 3 nb means sp3, 4 nb still sp3
        8:{1:2, 2:3, 3:3, 4:3}, # O
        15:{2:2, 3:3, 4:3},     # P - like N
        16:{2:2, 3:3, 4:3},     # S
    }
    return d[element].get(len(neighbors(graph, index)))

def is_amide_n(index, graph, mode=-1):
    '''
    Returns true if the nitrogen atom at the given
    index is a nitrogen and is part of an amide.
    Carbamates and ureas are considered amides.

    mode:
    -1 - any amide
    0 - primary amide (CONH2)
    1 - secondary amide (CONHR)
    2 - tertiary amide (CONR2)
    '''
    if graph.nodes[index]['atomnos'] == 7:
        # index must be a nitrogen atom

        nb = neighbors(graph, index)
        nb_atomnos = [graph.nodes[j]['atomnos'] for j in nb]

        if mode != -1:
            if nb_atomnos.count(1) != (2,1,0)[mode]:
                # primary amides need to have 1H, secondary amides none
                return False

        for n in nb:
            if graph.nodes[n]['atomnos'] == 6:
            # there must be at least one carbon atom next to N

                nb_nb = neighbors(graph, n)
                if len(nb_nb) == 3:
                # bonded to three atoms

                    nb_nb_sym = [graph.nodes[i]['atomnos'] for i in nb_nb]
                    if 8 in nb_nb_sym:
                        return True
                        # and at least one of them has to be an oxygen
    return False

def is_ester_o(index, graph):
    '''
    Returns true if the atom at the given
    index is an oxygen and is part of an ester.
    Carbamates and carbonates return True,
    Carboxylic acids return False.
    '''
    if graph.nodes[index]['atomnos'] == 8:
        nb = neighbors(graph, index)
        if 1 not in nb:
            for n in nb:
                if graph.nodes[n]['atomnos'] == 6:
                    nb_nb = neighbors(graph, n)
                    if len(nb_nb) == 3:
                        nb_nb_sym = [graph.nodes[i]['atomnos'] for i in nb_nb]
                        if nb_nb_sym.count(8) > 1:
                            return True
    return False

def is_phenyl(coords):
    '''
    :params coords: six coordinates of C/N atoms
    :return tuple: bool indicating if the six atoms look like part of a
                   phenyl/naphtyl/pyridine system, coordinates for the center of that ring

    NOTE: quinones would show as aromatic: it is okay, since they can do π-stacking as well.
    '''

    if np.max(all_dists(coords, coords)) > 3:
        return False
    # if any atomic couple is more than 3 A away from each other, this is not a Ph

    threshold_delta = 1 - np.cos(10 * np.pi/180)
    flat_delta = 1 - np.abs(np.cos(dihedral(coords[[0,1,2,3]]) * np.pi/180))

    if flat_delta < threshold_delta:
        flat_delta = 1 - np.abs(np.cos(dihedral(coords[[0,1,2,3]]) * np.pi/180))
        if flat_delta < threshold_delta:
            # print('phenyl center at', np.mean(coords, axis=0))
            return True
    
    return False

def get_phenyls(coords, atomnos):
    '''
    returns a (n, 6, 3) array where the first
    dimension is the aromatic rings detected
    '''
    if len(atomnos) < 6:
        return np.array([])

    output = []

    c_n_indexes = np.fromiter((i for i, a in enumerate(atomnos) if a in (6,7)), dtype=atomnos.dtype)
    comb = combinations(c_n_indexes, 6)

    for c in comb:
        mask = np.fromiter((i in c for i in range(len(atomnos))), dtype=bool)
        coords_ = coords[mask]
        if is_phenyl(coords_):
            output.append(coords_)

    return np.array(output)

def findPaths(G, u, n, excludeSet = None):
    '''
    Recursively find all paths of a NetworkX
    graph G with length = n, starting from node u
    '''
    if excludeSet is None:
        excludeSet = set([u])

    else:
        excludeSet.add(u)

    if n == 0:
        return [[u]]

    paths = [[u]+path for neighbor in G.neighbors(u) if neighbor not in excludeSet for path in findPaths(G,neighbor,n-1,excludeSet)]
    excludeSet.remove(u)

    return paths

@nb.njit
def torsion_comp_check(coords, torsion, mask, thresh=1.5, max_clashes=0) -> bool:
    '''
    coords: 3D molecule coordinates
    mask: 1D boolean array with the mask torsion
    thresh: threshold value for when two atoms are considered clashing
    max_clashes: maximum number of clashes to pass a structure
    returns True if the molecule shows less than max_clashes
    '''
    _, i2, i3, _ = torsion


    antimask = ~mask
    antimask[i2] = False
    antimask[i3] = False
    # making sure the i2-i3 bond is not included in the clashes

    m1 = coords[mask]
    m2 = coords[antimask]
    # fragment identification by boolean masking

    return 0 if np.count_nonzero(all_dists(m2,m1) < thresh) > max_clashes else 1

def cartesian_product(*arrays):
    return np.stack(np.meshgrid(*arrays), -1).reshape(-1, len(arrays))

def sanitize_conformer(coords, torsions, graph, thresh=1.5):
    '''
    Random dihedral rotations - quickly generate n_out conformers

    n_out: number of output structures
    max_tries: if n_out conformers are not generated after these number of tries, stop trying
    rotations: number of dihedrals to rotate per conformer. If none, all will be rotated
    '''
    rotations = 0
    while True:
    
        torsions_to_fix = []
        for torsion in torsions:
            mask = _get_rotation_mask(graph, torsion.torsion)
            if not torsion_comp_check(coords, torsion=torsion.torsion, mask=mask, thresh=thresh):
                torsions_to_fix.append(torsion)

        if not torsions_to_fix:
            return coords

        # for t, torsion in enumerate(torsions_to_fix):
        for t, torsion in enumerate(torsions):

            # for every angle we have to rotate, turn it 10 degrees until sane
            mask = _get_rotation_mask(graph, torsion.torsion)

            for _ in range(8):
                if torsion_comp_check(coords, torsion=torsion.torsion, mask=mask, thresh=thresh):
                    break
                coords = rotate_dihedral(coords, torsion.torsion, 10, mask=mask)
                rotations += 1
                print(f"Fixing conformation: torsion {t}, {rotations} turn", end='\r')

# def openbabel_opt(      structure,
#                         atomnos,
#                         constrained_indexes,
#                         constrained_distances=None,
#                         tight_constraint=True,
#                         method='MMFF94',
#                         nsteps=1000,
#                         title='temp_ob',
#                         **kwargs,
#                     ):
#         '''
#         tight_constraint: False uses the native implementation,
#                           True uses a more accurate recursive one 
#         return : MM-optimized structure (UFF/MMFF94)
#         '''

#         assert method in ('UFF', 'MMFF94', 'Ghemical', 'GAFF'), 'OpenBabel implements only the UFF, MMFF94, Ghemical and GAFF Force Fields.'

#         # If we have any target distance to impose,
#         # the most accurate way to do it is to manually
#         # move the second atom and then freeze both atom
#         # in place during optimization. If we would have
#         # to move the second atom too much we do that in
#         # small steps of 0.2 A, recursively, to avoid having
#         # openbabel come up with weird bonding topologies,
#         # ending in scrambling.

#         if constrained_distances is not None and tight_constraint:
#             for target_d, (a, b) in zip(constrained_distances, constrained_indexes):
#                 d = norm_of(structure[b] - structure[a])
#                 delta = d - target_d

#                 if abs(delta) > 0.2:
#                     sign = (d > target_d)
#                     recursive_c_d = [d + 0.2 * sign for d in constrained_distances]

#                     structure = openbabel_opt(
#                                                     structure,
#                                                     atomnos,
#                                                     constrained_indexes,
#                                                     constrained_distances=recursive_c_d,
#                                                     tight_constraint=True, 
#                                                     method=method,
#                                                     nsteps=nsteps,
#                                                     title=title,
#                                                     **kwargs,
#                                                 )

#                 d = norm_of(structure[b] - structure[a])
#                 delta = d - target_d
#                 structure[b] -= norm(structure[b] - structure[a]) * delta

#         filename=f'{title}_in.xyz'

#         with open(filename, 'w') as f:
#             write_xyz(structure, atomnos, f)
#         # input()
#         outname = f'{title}_out.xyz'

#         # Standard openbabel molecule load
#         conv = ob.OBConversion()
#         conv.SetInAndOutFormats('xyz','xyz')
#         mol = ob.OBMol()
#         more = conv.ReadFile(mol, filename)
#         i = 0

#         # Define constraints
#         constraints = ob.OBFFConstraints()

#         for i, (a, b) in enumerate(constrained_indexes):

#             # Adding a distance constraint does not lead to accurate results,
#             # so the backup solution is to freeze the atoms in place
#             if tight_constraint:
#                 constraints.AddAtomConstraint(int(a+1))
#                 constraints.AddAtomConstraint(int(b+1))

#             else:
#                 if constrained_distances is None:
#                     first_atom = mol.GetAtom(int(a+1))
#                     length = first_atom.GetDistance(int(b+1))
#                 else:
#                     length = constrained_distances[i]
                
#                 constraints.AddDistanceConstraint(int(a+1), int(b+1), length)       # Angstroms

#                 # constraints.AddAngleConstraint(1, 2, 3, 120.0)      # Degrees
#                 # constraints.AddTorsionConstraint(1, 2, 3, 4, 180.0) # Degrees

#         # Setup the force field with the constraints
#         forcefield = ob.OBForceField.FindForceField(method)
#         forcefield.Setup(mol, constraints)

#         # Set the strictness of the constraint
#         forcefield.SetConstraints(constraints)

#         # Do a nsteps conjugate gradient minimization
#         # (or less if converges) and save the coordinates to mol.
#         forcefield.ConjugateGradients(nsteps)
#         forcefield.GetCoordinates(mol)
#         energy = forcefield.Energy() * 0.2390057361376673 # kJ/mol to kcal/mol

#         # Write the mol to a file
#         conv.WriteFile(mol,outname)
#         conv.CloseOutFile()

#         opt_coords = read_xyz(outname).atomcoords[0]

#         # clean_directory((f'{title}_in.xyz', f'{title}_out.xyz'))
        
#         return opt_coords
        
def clean_directory(to_remove=None):

    if to_remove:
        for name in to_remove:
            os.remove(name)

    for f in os.listdir():
        if f.split('.')[0] == 'temp':
            os.remove(f)
        elif f.startswith('temp_'):
            os.remove(f)

def graphize(coords, atomnos, mask=None):
    '''
    :params coords: atomic coordinates as 3D vectors
    :params atomnos: atomic numbers as a list
    :params mask: bool array, with False for atoms
                  to be excluded in the bond evaluation
    :return connectivity graph
    '''

    mask = np.array([True for _ in atomnos], dtype=bool) if mask is None else mask

    matrix = np.zeros((len(coords),len(coords)))
    for i, _ in enumerate(coords):
        for j in range(i,len(coords)):
            if mask[i] and mask[j]:
                if norm_of(coords[i]-coords[j]) < d_min_bond(atomnos[i], atomnos[j]):
                    matrix[i][j] = 1

    graph = nx.from_numpy_matrix(matrix)
    nx.set_node_attributes(graph, dict(enumerate(atomnos)), 'atomnos')

    return graph