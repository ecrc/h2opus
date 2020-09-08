#ifndef __HORTHOG_MARSHAL_H__
#define __HORTHOG_MARSHAL_H__

#include <h2opus/core/h2opus_handle.h>

// Routines to marshal pointers to project a coupling node using the corresponding node of
// the projection tree produced from the upsweep: S_{ts} = Tu_{t} S_{ts} Tv^T_{s}
#include <h2opus/marshal/horthog_project.cuh>

// Routine to marshal pointers to produce stacked TE nodes
// for example with two children we would have TE = [T_1 * E_1; T_2 * E_2]
#include <h2opus/marshal/horthog_upsweep.cuh>

// Routine to marshal pointers to let us split the TE matrix into sub-matrices
// to copy them into the transfer matrices
#include <h2opus/marshal/horthog_copy_block.cuh>

#endif
