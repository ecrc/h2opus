#include <h2opus/core/h2opus_defs.h>
#include <h2opus/core/thrust_runtime.h>

#include <thrust/for_each.h>
//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Thrust unary function
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T> struct HorthogUpsweep_Functor
{
    T *T_base, *E_base, *TE_base;
    T **ptr_T, **ptr_E, **ptr_TE;
    int child_start, parent_start, max_children;
    int child_new_rank, child_rank, parent_rank;
    int *head, *next;

    HorthogUpsweep_Functor(T *T_base, T *E_base, T *TE_base, T **ptr_T, T **ptr_E, T **ptr_TE, int child_new_rank,
                           int child_rank, int parent_rank, int child_start, int parent_start, int max_children,
                           int *head, int *next)
    {
        this->T_base = T_base;
        this->E_base = E_base;
        this->TE_base = TE_base;
        this->ptr_T = ptr_T;
        this->ptr_E = ptr_E;
        this->ptr_TE = ptr_TE;

        this->child_new_rank = child_new_rank;
        this->child_rank = child_rank;
        this->parent_rank = parent_rank;
        this->child_start = child_start;
        this->parent_start = parent_start;
        this->max_children = max_children;

        this->head = head;
        this->next = next;
    }

    __host__ __device__ void operator()(const unsigned int &node_index) const
    {
        T *TE_node = TE_base + node_index * max_children * child_new_rank * parent_rank;

        int child_index = head[node_index + parent_start];
        for (int child = 0; child < max_children; child++)
        {
            if (child_index != H2OPUS_EMPTY_NODE)
            {
                int child_level_index = child_index - child_start;

                ptr_TE[child_level_index] = TE_node + child * child_new_rank;
                ptr_T[child_level_index] = T_base + child_level_index * child_new_rank * child_rank;
                ptr_E[child_level_index] = E_base + child_level_index * child_rank * parent_rank;

                child_index = next[child_index];
            }
        }
    }
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////
// Driver routines
//////////////////////////////////////////////////////////////////////////////////////////////////////////
template <class T, int hw>
inline void horthog_upsweep_batch_marshal(T *T_base, T *E_base, T *TE_base, T **ptr_T, T **ptr_E, T **ptr_TE,
                                          int child_new_rank, int child_rank, int parent_rank, int child_start,
                                          int parent_start, int max_children, int *head, int *next, int num_parents,
                                          h2opusComputeStream_t stream)
{
    HorthogUpsweep_Functor<T> upsweep_functor(T_base, E_base, TE_base, ptr_T, ptr_E, ptr_TE, child_new_rank, child_rank,
                                              parent_rank, child_start, parent_start, max_children, head, next);

    thrust::for_each(ThrustRuntime<hw>::get(stream), thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(num_parents), upsweep_functor);
}
