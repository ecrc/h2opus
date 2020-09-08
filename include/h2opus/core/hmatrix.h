#ifndef __HAMTRIX_H__
#define __HAMTRIX_H__

#include <h2opus/core/basis_tree.h>
#include <h2opus/core/hnode_tree.h>

template <int hw> struct THMatrix
{
    typedef typename VectorContainer<hw, H2Opus_Real>::type RealVector;
    typedef typename VectorContainer<hw, int>::type IntVector;
    typedef typename TTreeContainer<RealVector>::type TreeContainer;
    typedef RealVector RealVec;
    typedef IntVector IntVec;

    THNodeTree<hw> hnodes;
    TBasisTree<hw> u_basis_tree, v_basis_tree;
    int n;
    bool sym;

    THMatrix(int n, bool symmetric = false)
    {
        this->n = n;
        this->sym = symmetric;
    }

    template <int other_hw> THMatrix(const THMatrix<other_hw> &h)
    {
        init(h);
    }

    template <int other_hw> THMatrix &operator=(const THMatrix<other_hw> &h)
    {
        init(h);
        return *this;
    }

    // Get memory consumption in GB
    H2Opus_Real getMemoryUsage()
    {
        return u_basis_tree.getMemoryUsage() + v_basis_tree.getMemoryUsage() + hnodes.getMemoryUsage();
    }

    H2Opus_Real getDenseMemoryUsage()
    {
        return hnodes.getDenseMemoryUsage();
    }

    H2Opus_Real getLowRankMemoryUsage()
    {
        return u_basis_tree.getMemoryUsage() + v_basis_tree.getMemoryUsage() + hnodes.getLowRankMemoryUsage();
    }

    // Clear only the matrix data, leaving structure data intact
    void clearData()
    {
        u_basis_tree.clearData();
        v_basis_tree.clearData();
        hnodes.clearData();
    }

  private:
    template <int other_hw> void init(const THMatrix<other_hw> &h)
    {
        this->n = h.n;
        this->sym = h.sym;

        this->hnodes = h.hnodes;
        this->u_basis_tree = h.u_basis_tree;
        this->v_basis_tree = h.v_basis_tree;
    }
};

#ifdef H2OPUS_USE_GPU
typedef THMatrix<H2OPUS_HWTYPE_GPU> HMatrix_GPU;
typedef HMatrix_GPU::TreeContainer TreeContainer_GPU;
typedef HMatrix_GPU::RealVector RealVector_GPU;
typedef HMatrix_GPU::IntVector IntVector_GPU;
#endif

typedef THMatrix<H2OPUS_HWTYPE_CPU> HMatrix;
typedef HMatrix::TreeContainer TreeContainer;
typedef HMatrix::RealVector RealVector;
typedef HMatrix::IntVector IntVector;

#endif
