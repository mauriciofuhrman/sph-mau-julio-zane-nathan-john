#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vec3.hpp"
#include "zmorton.hpp"
#include <vector>

#include "params.hpp"
#include "state.hpp"
#include "interact.hpp"
#include "binhash.hpp"

/* Define this to use the bucketing version of the code */
#define USE_BUCKETING

/* Define to use the parallel version of the code*/
#define USE_PARALLEL

/*@T
 * \subsection{Density computations}
 * 
 * The formula for density is
 * \[
 *   \rho_i = \sum_j m_j W_{p6}(r_i-r_j,h)
 *          = \frac{315 m}{64 \pi h^9} \sum_{j \in N_i} (h^2 - r^2)^3.
 * \]
 * We search for neighbors of node $i$ by checking every particle,
 * which is not very efficient.  We do at least take advange of
 * the symmetry of the update ($i$ contributes to $j$ in the same
 * way that $j$ contributes to $i$).
 *@c*/

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "vec3.hpp"
#include "zmorton.hpp"
#include <vector>
#include "params.hpp"
#include "state.hpp"
#include "interact.hpp"
#include "binhash.hpp"
#include <omp.h>

/* Define this to use the bucketing version of the code */
#define USE_BUCKETING

/* Define to use the parallel version of the code*/
#define USE_PARALLEL

inline
void update_density(particle_t* pi, particle_t* pj, float h2, float C, float* i_rho, float* j_rho)
{
    float r2 = vec3_dist2(pi->x, pj->x);
    float z  = h2 - r2;
    if (z > 0) {
        float rho_ij = C * z * z * z;
        *i_rho += rho_ij;
        *j_rho += rho_ij;
    }
}

void compute_density(sim_state_t* s, sim_param_t* params)
{
    int n = s->n;
    particle_t* p = s->part;
    particle_t** hash = s->hash;

    float h  = params->h;
    float h2 = h * h;
    float h3 = h2 * h;
    float h9 = h3 * h3 * h3;
    float C  = (315.0 / 64.0 / M_PI) * s->mass / h9;
    float base_rho = (315.0 / 64.0 / M_PI) * s->mass / h3;

    // Initialize base densities
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        p[i].rho = base_rho;
    }

    // Get total number of threads once
    const int num_threads = omp_get_max_threads();
    
    // Create per-thread storage outside parallel region
    std::vector<std::vector<float>> thread_local_density(num_threads, std::vector<float>(n, 0.0f));

    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        float* my_local_density = thread_local_density[thread_id].data();

        // Process spatial bins in parallel
        #pragma omp for schedule(dynamic, 64)
        for (unsigned bin = 0; bin < HASH_SIZE; ++bin) {
            particle_t* pi = hash[bin];
            
            while (pi) {
                unsigned buckets[MAX_NBR_BINS];
                unsigned num_buckets = particle_neighborhood(buckets, pi, h);
                const int pi_idx = pi - p;  // Get particle index
                
                for (unsigned b = 0; b < num_buckets; ++b) {
                    unsigned bj = buckets[b];
                    if (bj < bin) continue; // Process each pair once
                    
                    particle_t* pj = hash[bj];
                    while (pj) {
                        if (pi == pj) {
                            pj = pj->next;
                            continue;
                        }
                        
                        // If in same bin, only process forward direction
                        if (bin == bj && pj <= pi) {
                            pj = pj->next;
                            continue;
                        }

                        const int pj_idx = pj - p;  // Get particle index
                        
                        float r2 = vec3_dist2(pi->x, pj->x);
                        float z  = h2 - r2;
                        if (z > 0) {
                            float rho_ij = C * z * z * z;
                            // Add to thread-local storage for both particles
                            my_local_density[pi_idx] += rho_ij;
                            my_local_density[pj_idx] += rho_ij;
                        }
                        
                        pj = pj->next;
                    }
                }
                pi = pi->next;
            }
        }
    }

    // Final reduction step - iterate over threads first, then particles
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < num_threads; ++t) {
        const float* thread_density = thread_local_density[t].data();
        for (int i = 0; i < n; ++i) {
            #pragma omp atomic
            p[i].rho += thread_density[i];
        }
    }
}
/*@T
 * \subsection{Computing forces}
 * 
 * The acceleration is computed by the rule
 * \[
 *   \bfa_i = \frac{1}{\rho_i} \sum_{j \in N_i} 
 *     \bff_{ij}^{\mathrm{interact}} + \bfg,
 * \]
 * where the pair interaction formula is as previously described.
 * Like [[compute_density]], the [[compute_accel]] routine takes
 * advantage of the symmetry of the interaction forces
 * ($\bff_{ij}^{\mathrm{interact}} = -\bff_{ji}^{\mathrm{interact}}$)
 * but it does a very expensive brute force search for neighbors.
 *@c*/

#ifdef USE_PARALLEL
inline
void update_forces(particle_t* pi, particle_t* pj, float h2,
                   float rho0, float C0, float Cp, float Cv)
{
    float dx[3];
    vec3_diff(dx, pi->x, pj->x);
    float r2 = vec3_len2(dx);
    if (r2 < h2) {
        const float rhoi = pi->rho;
        const float rhoj = pj->rho;
        float q = sqrt(r2/h2);
        float u = 1-q;
        float w0 = C0 * u/rhoi/rhoj;
        float wp = w0 * Cp * (rhoi+rhoj-2*rho0) * u/q;
        float wv = w0 * Cv;
        float dv[3];
        vec3_diff(dv, pi->v, pj->v);

        // only want to compute forces acting on pi, to prevent concurrent writes to shared data
        // Equal and opposite pressure forces
        vec3_saxpy(pi->a,  wp, dx);
        // vec3_saxpy(pj->a, -wp, dx);
        
        // // Equal and opposite viscosity forces
        vec3_saxpy(pi->a,  wv, dv);
        // vec3_saxpy(pj->a, -wv, dv);
    }
}
#else
inline
void update_forces(particle_t* pi, particle_t* pj, float h2,
                   float rho0, float C0, float Cp, float Cv)
{
    float dx[3];
    vec3_diff(dx, pi->x, pj->x);
    float r2 = vec3_len2(dx);
    if (r2 < h2) {
        const float rhoi = pi->rho;
        const float rhoj = pj->rho;
        float q = sqrt(r2/h2);
        float u = 1-q;
        float w0 = C0 * u/rhoi/rhoj;
        float wp = w0 * Cp * (rhoi+rhoj-2*rho0) * u/q;
        float wv = w0 * Cv;
        float dv[3];
        vec3_diff(dv, pi->v, pj->v);

        // Equal and opposite pressure forces
        vec3_saxpy(pi->a,  wp, dx);
        vec3_saxpy(pj->a, -wp, dx);
        
        // Equal and opposite viscosity forces
        vec3_saxpy(pi->a,  wv, dv);
        vec3_saxpy(pj->a, -wv, dv);
    }
}
#endif

void compute_accel(sim_state_t* state, sim_param_t* params)
{
    // Unpack basic parameters
    const float h    = params->h;
    const float rho0 = params->rho0;
    const float k    = params->k;
    const float mu   = params->mu;
    const float g    = params->g;
    const float mass = state->mass;
    const float h2   = h*h;

    // Unpack system state
    particle_t* p = state->part;
    particle_t** hash = state->hash;
    int n = state->n;

    // Rehash the particles
    hash_particles(state, h);

    // Compute density and color
    compute_density(state, params);

    // Constants for interaction term
    float C0 = 45 * mass / M_PI / ((h2)*(h2)*h);
    float Cp = k/2;
    float Cv = -mu;

    // Get total number of threads once
    const int num_threads = omp_get_max_threads();
    
    // Create per-thread storage for acceleration contributions
    struct ThreadAccel {
        float x, y, z;
        ThreadAccel() : x(0), y(0), z(0) {}
    };
    std::vector<std::vector<ThreadAccel>> thread_local_accel(num_threads, std::vector<ThreadAccel>(n));

    // Initialize with gravity in parallel
    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        auto& my_local_accel = thread_local_accel[thread_id];
        
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            my_local_accel[i].x = 0;
            my_local_accel[i].y = -g;
            my_local_accel[i].z = 0;
        }
    }

    // Compute forces using thread-local storage
    #pragma omp parallel
    {
        const int thread_id = omp_get_thread_num();
        auto& my_local_accel = thread_local_accel[thread_id];

        #pragma omp for schedule(dynamic, 64)
        for (unsigned bin = 0; bin < HASH_SIZE; ++bin) {
            particle_t* pi = hash[bin];
            
            while (pi) {
                unsigned buckets[MAX_NBR_BINS];
                unsigned num_buckets = particle_neighborhood(buckets, pi, h);
                const int pi_idx = pi - p;

                for (unsigned b = 0; b < num_buckets; ++b) {
                    unsigned bj = buckets[b];
                    if (bj < bin) continue; // Process each pair once
                    
                    particle_t* pj = hash[bj];
                    while (pj) {
                        if (pi == pj) {
                            pj = pj->next;
                            continue;
                        }
                        
                        // If in same bin, only process forward direction
                        if (bin == bj && pj <= pi) {
                            pj = pj->next;
                            continue;
                        }

                        const int pj_idx = pj - p;
                        
                        float dx[3];
                        vec3_diff(dx, pi->x, pj->x);
                        float r2 = vec3_len2(dx);
                        
                        if (r2 < h2) {
                            const float rhoi = pi->rho;
                            const float rhoj = pj->rho;
                            float q = sqrt(r2/h2);
                            float u = 1-q;
                            float w0 = C0 * u/rhoi/rhoj;
                            float wp = w0 * Cp * (rhoi+rhoj-2*rho0) * u/q;
                            float wv = w0 * Cv;
                            float dv[3];
                            vec3_diff(dv, pi->v, pj->v);

                            // Add pressure force contributions
                            my_local_accel[pi_idx].x += wp * dx[0];
                            my_local_accel[pi_idx].y += wp * dx[1];
                            my_local_accel[pi_idx].z += wp * dx[2];
                            my_local_accel[pj_idx].x -= wp * dx[0];
                            my_local_accel[pj_idx].y -= wp * dx[1];
                            my_local_accel[pj_idx].z -= wp * dx[2];

                            // Add viscosity force contributions
                            my_local_accel[pi_idx].x += wv * dv[0];
                            my_local_accel[pi_idx].y += wv * dv[1];
                            my_local_accel[pi_idx].z += wv * dv[2];
                            my_local_accel[pj_idx].x -= wv * dv[0];
                            my_local_accel[pj_idx].y -= wv * dv[1];
                            my_local_accel[pj_idx].z -= wv * dv[2];
                        }
                        
                        pj = pj->next;
                    }
                }
                pi = pi->next;
            }
        }
    }

    // Final reduction step - combine all thread contributions
    // Iterate over threads in outer loop for better cache locality
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < num_threads; ++t) {
        const auto& thread_accel = thread_local_accel[t];
        for (int i = 0; i < n; ++i) {
            #pragma omp atomic
            p[i].a[0] += thread_accel[i].x;
            #pragma omp atomic
            p[i].a[1] += thread_accel[i].y;
            #pragma omp atomic
            p[i].a[2] += thread_accel[i].z;
        }
    }
}