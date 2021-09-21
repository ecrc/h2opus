#ifndef __H2OPUS_GEOMETRIC_ADMISSIBILITY_H__
#define __H2OPUS_GEOMETRIC_ADMISSIBILITY_H__

#include <h2opus/util/kdtree.h>

template <class T, int hw> class TH2OpusAdmissibility
{
  public:
    virtual bool operator()(TH2OpusKDTree<T, hw> *kd_tree, int node_index_u, int node_index_v) = 0;
};

template <class T, int hw> class TH2OpusSeperateAdmissibility : public TH2OpusAdmissibility<T, hw>
{
  public:
    bool operator()(TH2OpusKDTree<T, hw> *kd_tree, int node_index_u, int node_index_v)
    {
        int u_start, u_end, v_start, v_end;
        kd_tree->getNodeLimits(node_index_u, u_start, u_end);
        kd_tree->getNodeLimits(node_index_v, v_start, v_end);

        return (u_start > v_end || v_start > u_end);
    }
};

template <class T, int hw> class TH2OpusBoxEdgeAdmissibility : public TH2OpusAdmissibility<T, hw>
{
  private:
    T eta;

  public:
    bool operator()(TH2OpusKDTree<T, hw> *kd_tree, int node_index_u, int node_index_v)
    {
        T dist = h2opusBBoxDist<T, hw>(kd_tree, node_index_u, node_index_v);
        T diam_u = h2opusBBoxDiam<T, hw>(kd_tree, node_index_u);
        T diam_v = h2opusBBoxDiam<T, hw>(kd_tree, node_index_v);

        if (diam_u == 0 || diam_v == 0)
            return false;

        return (std::min(diam_u, diam_v) <= eta * dist);
    }

    TH2OpusBoxEdgeAdmissibility(T eta)
    {
        this->eta = eta;
    }
};

template <class T, int hw> class TH2OpusBoxCenterAdmissibility : public TH2OpusAdmissibility<T, hw>
{
  private:
    T eta;

  public:
    bool operator()(TH2OpusKDTree<T, hw> *kd_tree, int node_index_u, int node_index_v)
    {
        T dist = h2opusBBoxCenterDist<T, hw>(kd_tree, node_index_u, node_index_v);
        T diam_u = h2opusBBoxDiam<T, hw>(kd_tree, node_index_u);
        T diam_v = h2opusBBoxDiam<T, hw>(kd_tree, node_index_v);

        if (diam_u == 0 || diam_v == 0)
            return false;

        return (0.5 * (diam_u + diam_v) <= eta * dist);
    }

    TH2OpusBoxCenterAdmissibility(T eta)
    {
        this->eta = eta;
    }
};

typedef TH2OpusAdmissibility<H2Opus_Real, H2OPUS_HWTYPE_CPU> H2OpusAdmissibility;
typedef TH2OpusSeperateAdmissibility<H2Opus_Real, H2OPUS_HWTYPE_CPU> H2OpusSeperateAdmissibility;
typedef TH2OpusBoxEdgeAdmissibility<H2Opus_Real, H2OPUS_HWTYPE_CPU> H2OpusBoxEdgeAdmissibility;
typedef TH2OpusBoxCenterAdmissibility<H2Opus_Real, H2OPUS_HWTYPE_CPU> H2OpusBoxCenterAdmissibility;

#endif
