#ifndef __HARA_MARSHAL_COMMON_H__
#define __HARA_MARSHAL_COMMON_H__

struct HARA_LowerBound_To_HNode_Functor
{
    int *update_morton_indexes, *lower_bounds, *node_morton_level_index;
    int hnode_start_index, num_hnodes;

    HARA_LowerBound_To_HNode_Functor(int *update_morton_indexes, int *lower_bounds, int *node_morton_level_index,
                                     int hnode_start_index, int num_hnodes)
    {
        this->update_morton_indexes = update_morton_indexes;
        this->lower_bounds = lower_bounds;
        this->node_morton_level_index = node_morton_level_index;
        this->hnode_start_index = hnode_start_index;
        this->num_hnodes = num_hnodes;
    }

    __host__ __device__ void operator()(const unsigned int &update_index) const
    {
        int key = update_morton_indexes[update_index];
        int lower_bound = lower_bounds[update_index];

        // Check if the key is outside the array bounds
        int hnode_index = -1;

        if (lower_bound < num_hnodes)
        {
            // Check if they key matches the lower bound value
            int hnode_morton_index = node_morton_level_index[hnode_start_index + lower_bound];

            // If we have a match, return the hnode index within the level
            if (key == hnode_morton_index)
                hnode_index = hnode_start_index + lower_bound;
        }

        // The lower bounds is updated to store the hnode index, since we're temporarily
        // using the update hnode index array to store the lower bounds
        lower_bounds[update_index] = hnode_index;
    }
};

#endif
