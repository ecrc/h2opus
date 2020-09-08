#ifndef __HCOMPRESS_MARSHAL_H__
#define __HCOMPRESS_MARSHAL_H__

#include <h2opus/core/h2opus_handle.h>

// Optimal basis generation marshaling routines
#include <h2opus/marshal/hcompress_basis_gen.cuh>

// Routine to marshal pointers to produce stacked TE nodes
// for example with two children we would have TE = [T_1 * E_1; T_2 * E_2]
#include <h2opus/marshal/hcompress_upsweep.cuh>

// Routine to marshal pointers to let us split the TE matrix into sub-matrices
// to copy them into the transfer matrices
#include <h2opus/marshal/hcompress_copy_block.cuh>

// Routines to marshal pointers to project a coupling node using the corresponding node of
// the projection tree produced from the upsweep: S_{ts} = Tu_{t} S_{ts} Tv^T_{s}
#include <h2opus/marshal/hcompress_project.cuh>

#endif
