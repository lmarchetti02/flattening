#include <math.h>
#include <stdlib.h>
#include <omp.h>

/**
 * Efficiently performs the interpolation loop. (See Python functions.)
 *
 * @param proj The pointer to the array of curved projections.
 * @param normalized_angles The pointer to the array of angles.
 * @param out The pointer to the array of flat projections.
 * @param num_proj The number of projections.
 * @param num_rows The number of detector rows.
 * @param orig_num_detectors The number of (curved) detector columns.
 * @param num_cols The number of (flat) detector columns(> curved).
 */
void interpolation_loop(
    const float *proj,               // input [num_proj, num_rows, orig_num_detectors]
    const double *normalized_angles, // input [num_cols]
    float *out,                      // output [num_proj, num_rows, num_cols]
    int num_proj,
    int num_rows,
    int orig_num_detectors,
    int num_cols)
{
// Parallelize outer loops over i and r
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < num_proj; i++)
    {
        for (int r = 0; r < num_rows; r++)
        {
            for (int j = 0; j < num_cols; j++)
            {
                double x = normalized_angles[j]; // pos on curved detector
                int idx = (int)floor(x);         // index of left neighbor
                double t = x - idx;              // distance from right neighbor

                // index of start of (i,r) row in in & out
                int proj_offset = (i * num_rows + r) * orig_num_detectors;
                int out_offset = (i * num_rows + r) * num_cols;

                // before first detector
                if (idx < 0)
                    out[out_offset + j] = proj[proj_offset];
                // after last detector
                else if (idx >= orig_num_detectors - 1)
                    out[out_offset + j] = proj[proj_offset + orig_num_detectors - 1];
                else
                {
                    float v0 = proj[proj_offset + idx];                     // value at left neighbor
                    float v1 = proj[proj_offset + idx + 1];                 // value at right neighbor
                    out[out_offset + j] = (float)((1.0 - t) * v0 + t * v1); // linear interpolation
                }
            }
        }
    }
}
