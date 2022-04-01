import ctypes
import numpy as np
import math


class H2Mat(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int),
        ("leaf_size", ctypes.c_int),
        ("depth", ctypes.c_int),
        ("precision", ctypes.c_int),
        ("index_map", ctypes.POINTER(ctypes.c_int)),
        # dense blocks
        ("num_dense_leaves", ctypes.c_int),
        ("dense_node_indexes", ctypes.POINTER(ctypes.c_int)),
        ("u_index", ctypes.POINTER(ctypes.c_int)),
        ("v_index", ctypes.POINTER(ctypes.c_int)),
        ("dense_leaf_mem", ctypes.POINTER(ctypes.c_float)),
        # lr blocks
        ("level_rank", ctypes.POINTER(ctypes.c_int)),
        ("num_lr_leaves", ctypes.POINTER(ctypes.c_int)),
        ("lr_node_indexes", ctypes.POINTER(ctypes.POINTER(ctypes.c_int))),
        ("lr_leaf_mem", ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
        # basis
        ("trans_dim", ctypes.POINTER(ctypes.c_int)),
        ("basis_mem", ctypes.POINTER(ctypes.c_float)),
        ("trans_mem", ctypes.POINTER(ctypes.POINTER(ctypes.c_float))),
    ]

    def __str__(self):
        data = ''
        for field_name, field_type in self._fields_:
            if field_type == ctypes.c_int:
                data += str(getattr(self, field_name)) + ', '
        return data


_lib_ch2opus = ctypes.CDLL('./libpyh2opus.so')
_lib_ch2opus.build_hmatrix.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_float)
_lib_ch2opus.build_hmatrix.restype = H2Mat


def btree_idx_to_level_idx(idx):
    level = math.floor(math.log2(idx + 1))
    level_index = idx - (2**level - 1)
    return level_index


def make_coo(nblocks, htree_index, basis_index):
    I = np.zeros(nblocks, dtype=np.int32)
    for i in range(nblocks):
        idx = htree_index[i]
        I[i] = btree_idx_to_level_idx(basis_index[idx])
    return I


def make_dense_blocks(nblocks, m, precision, buffer):
    if precision == 4:
        array_type = (ctypes.c_float * nblocks * m * m)
    elif precision == 8:
        array_type = (ctypes.c_double * nblocks * m * m)
    addr = ctypes.addressof(buffer.contents)
    # print('dense address', addr)
    A = np.ctypeslib.as_array(array_type.from_address(addr))
    A = A.reshape(nblocks, m, m)
    return A


def make_lr_indexes(depth, level_size, tree_indices, basis_index):
    lrI = []
    for l in range(depth):
        lrI.append(make_coo(level_size[l], tree_indices[l], basis_index))
    return lrI


def make_lr_blocks(depth, level_size, rank, precision, lr_buffers):
    if precision == 4:
        ftype = ctypes.c_float
    elif precision == 8:
        ftype = ctypes.c_double
    A = [np.empty(0, dtype=ftype) for l in range(depth)]
    # buffer_type = (ctypes.POINTER(ftype) * depth)
    # print(buffer_type, buffer_type._type_)
    # print(lr_buffers, lr_buffers.contents)
    # buffers = np.ctypeslib.as_array(buffer_type.from_address(ctypes.addressof(lr_buffers.contents)))
    for l in range(1, depth):
        k = rank[l]
        if level_size[l] != 0:
            array_type = (ftype * level_size[l] * k * k)
            # buffer = buffers[l]
            # print('lr buffer', l, lr_buffers[l])
            addr = ctypes.addressof(lr_buffers[l].contents)
            # print('lr address', addr)
            A_l = np.ctypeslib.as_array(array_type.from_address(addr))
            A[l] = A_l.reshape(level_size[l], k, k)
    return A


def make_basis(ftype, depth, m, k, buffer):
    num_leaves = 2**(depth - 1)
    array_type = (ftype * (num_leaves * m * k))
    addr = ctypes.addressof(buffer.contents)
    U = np.ctypeslib.as_array(array_type.from_address(addr))
    U = U.reshape(num_leaves, k, m)  # stored as transposed
    return U


def make_transfer(ftype, depth, trans_dim, buffers):
    E = [np.empty(0, dtype=ftype) for l in range(depth)]
    for l in range(depth):
        num_matrices = 2**l
        kc = trans_dim[l + 1]
        kp = trans_dim[l]
        array_type = (ftype * (num_matrices * kc * kp))
        addr = ctypes.addressof(buffers[l].contents)
        E_l = np.ctypeslib.as_array(array_type.from_address(addr))
        E[l] = E_l.reshape(num_matrices, kp, kc)  # E stored in transposed form
    return E


def export_vals(nblocks, m, precision, buffer):
    if precision == 4:
        array_type = (ctypes.c_float * nblocks * m * m)
    elif precision == 8:
        array_type = (ctypes.c_double * nblocks * m * m)
    addr = ctypes.addressof(buffer.contents)
    # print('dense address', addr)
    A = np.ctypeslib.as_array(array_type.from_address(addr))
    A = A.reshape(nblocks, m, m)
    return A


def make_coo_from_csr(n, ii, jj):
    nnz = ii[n]
    I = np.empty(nnz, dtype=np.int32)
    J = np.empty(nnz, dtype=np.int32)
    c = 0
    for i in range(n):
        for j in range(ii[i], ii[i + 1]):
            I[c] = ii[i]
            J[c] = jj[j]
            c = c + 1
    return I, J


def build_hmatrix(gx, gy, m, d, eta):
    result = _lib_ch2opus.build_hmatrix(gx, gy, m, d, eta)
    ftype = ctypes.c_float if result.precision == 4 else ctypes.c_double

    result.dI = make_coo(result.num_dense_leaves, result.dense_node_indexes,
                         result.u_index)
    result.dJ = make_coo(result.num_dense_leaves, result.dense_node_indexes,
                         result.v_index)
    result.dA = make_dense_blocks(result.num_dense_leaves, result.leaf_size,
                                  result.precision, result.dense_leaf_mem)

    result.lrI = make_lr_indexes(result.depth, result.num_lr_leaves,
                                 result.lr_node_indexes, result.u_index)
    result.lrJ = make_lr_indexes(result.depth, result.num_lr_leaves,
                                 result.lr_node_indexes, result.v_index)
    result.lrA = make_lr_blocks(result.depth, result.num_lr_leaves,
                                result.level_rank, result.precision,
                                result.lr_leaf_mem)

    result.U = make_basis(ftype, result.depth, result.leaf_size,
                          result.level_rank[result.depth - 1], result.basis_mem)
    result.E = make_transfer(ftype, result.depth, result.trans_dim,
                             result.trans_mem)
    return result


# def view1(nd, node_index, basis_index):
#     for i in range(nd):
#         idx = node_index[i]
#         print(i, idx, tree_idx_to_level_idx(basis_index[idx]))
# view1(result.num_dense_leaves, result.dense_node_indexes, result.u_index)
