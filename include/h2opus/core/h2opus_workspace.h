#ifndef __H2OPUS_WORKSPACE_H__
#define __H2OPUS_WORKSPACE_H__

#include <h2opusconf.h>

#ifdef H2OPUS_USE_GPU
#include <h2opus/util/gpu_err_check.h>
#endif

struct H2OpusWorkspaceState
{
    size_t d_data_bytes, d_ptrs_bytes;
    size_t h_data_bytes, h_ptrs_bytes;

    H2OpusWorkspaceState()
    {
        d_data_bytes = d_ptrs_bytes = 0;
        h_data_bytes = h_ptrs_bytes = 0;
    }

    void addHostBytes(size_t data, size_t ptrs)
    {
        setHostBytes(h_data_bytes + data, h_ptrs_bytes + ptrs);
    }
#ifdef H2OPUS_USE_GPU
    void addDeviceBytes(size_t data, size_t ptrs)
    {
        setDeviceBytes(d_data_bytes + data, d_ptrs_bytes + ptrs);
    }

    void setDeviceBytes(size_t data, size_t ptrs)
    {
        d_data_bytes = data;
        d_ptrs_bytes = ptrs;
    }
#endif
    void setHostBytes(size_t data, size_t ptrs)
    {
        h_data_bytes = data;
        h_ptrs_bytes = ptrs;
    }

    void setBytes(size_t data, size_t ptrs, int hw)
    {
        if (hw == H2OPUS_HWTYPE_GPU)
        {
#ifdef H2OPUS_USE_GPU
            setDeviceBytes(data, ptrs);
#else
            assert(false);
#endif
        }
        else
            setHostBytes(data, ptrs);
    }

    void getBytes(size_t &data, size_t &ptrs, int hw)
    {
        if (hw == H2OPUS_HWTYPE_GPU)
        {
#ifdef H2OPUS_USE_GPU
            data = d_data_bytes;
            ptrs = d_ptrs_bytes;
#else
            assert(false);
#endif
        }
        else
        {
            data = h_data_bytes;
            ptrs = h_ptrs_bytes;
        }
    }
};

inline bool operator<(const H2OpusWorkspaceState &l, const H2OpusWorkspaceState &r)
{
    return (l.d_data_bytes < r.d_data_bytes || l.d_ptrs_bytes < r.d_ptrs_bytes || l.h_data_bytes < r.h_data_bytes ||
            l.h_ptrs_bytes < r.h_ptrs_bytes);
}

struct H2OpusWorkspace
{
  private:
    H2OpusWorkspaceState allocated_ws_state;
    void *h_data, *h_ptrs;
#ifdef H2OPUS_USE_GPU
    void *d_data, *d_ptrs;
#endif

  public:
    H2OpusWorkspace()
    {
        h_data = h_ptrs = NULL;
#ifdef H2OPUS_USE_GPU
        d_data = d_ptrs = NULL;
#endif
    }

    ~H2OpusWorkspace()
    {
#ifdef H2OPUS_USE_GPU
        if (d_data)
            gpuErrchk(cudaFree(d_data));
        if (d_ptrs)
            gpuErrchk(cudaFree(d_ptrs));
#endif
        if (h_data)
            free(h_data);
        if (h_ptrs)
            free(h_ptrs);
    }

    H2OpusWorkspaceState getWorkspaceState()
    {
        return allocated_ws_state;
    }

    void setWorkspaceState(H2OpusWorkspaceState &ws_state)
    {
#ifdef H2OPUS_USE_GPU
        if (ws_state.d_data_bytes != 0 && ws_state.d_data_bytes > allocated_ws_state.d_data_bytes)
        {
            if (d_data)
                gpuErrchk(cudaFree(d_data));
            gpuErrchk(cudaMalloc(&d_data, ws_state.d_data_bytes));

            allocated_ws_state.d_data_bytes = ws_state.d_data_bytes;
        }

        if (ws_state.d_ptrs_bytes != 0 && ws_state.d_ptrs_bytes > allocated_ws_state.d_ptrs_bytes)
        {
            if (d_ptrs)
                gpuErrchk(cudaFree(d_ptrs));
            gpuErrchk(cudaMalloc(&d_ptrs, ws_state.d_ptrs_bytes));

            allocated_ws_state.d_ptrs_bytes = ws_state.d_ptrs_bytes;
        }
#endif
        if (ws_state.h_data_bytes != 0 && ws_state.h_data_bytes > allocated_ws_state.h_data_bytes)
        {
            if (h_data)
                free(h_data);
            h_data = malloc(ws_state.h_data_bytes);
            assert(h_data);

            allocated_ws_state.h_data_bytes = ws_state.h_data_bytes;
        }

        if (ws_state.h_data_bytes != 0 && ws_state.h_ptrs_bytes > allocated_ws_state.h_ptrs_bytes)
        {
            if (h_ptrs)
                free(h_ptrs);
            h_ptrs = malloc(ws_state.h_ptrs_bytes);
            assert(h_ptrs);

            allocated_ws_state.h_ptrs_bytes = ws_state.h_ptrs_bytes;
        }
    }

#ifdef H2OPUS_USE_GPU
    void *getDeviceData()
    {
        return d_data;
    }
    void *getDevicePtrs()
    {
        return d_ptrs;
    }
#endif
    void *getHostData()
    {
        return h_data;
    }
    void *getHostPtrs()
    {
        return h_ptrs;
    }

    void *getPtrs(int hw)
    {
        if (hw == H2OPUS_HWTYPE_GPU)
        {
#ifdef H2OPUS_USE_GPU
            return d_ptrs;
#else
            assert(false);
            return NULL;
#endif
        }
        else
            return h_ptrs;
    }

    void *getData(int hw)
    {
        if (hw == H2OPUS_HWTYPE_GPU)
        {
#ifdef H2OPUS_USE_GPU
            return d_data;
#else
            assert(false);
            return NULL;
#endif
        }
        else
            return h_data;
    }
};

typedef struct H2OpusWorkspace *h2opusWorkspace_t;

#endif
