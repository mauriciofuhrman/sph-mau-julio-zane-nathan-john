#include <string.h>

#include "zmorton.hpp"
#include "binhash.hpp"

/*@q
 * ====================================================================
 */

/*@T
 * \subsection{Spatial hashing implementation}
 * 
 * In the current implementation, we assume [[HASH_DIM]] is $2^b$,
 * so that computing a bitwise of an integer with [[HASH_DIM]] extracts
 * the $b$ lowest-order bits.  We could make [[HASH_DIM]] be something
 * other than a power of two, but we would then need to compute an integer
 * modulus or something of that sort.
 * 
 *@c*/

#define HASH_MASK (HASH_DIM-1)

unsigned particle_bucket(particle_t* p, float h)
{
    unsigned ix = p->x[0]/h;
    unsigned iy = p->x[1]/h;
    unsigned iz = p->x[2]/h;
    return zm_encode(ix & HASH_MASK, iy & HASH_MASK, iz & HASH_MASK);
}

unsigned particle_neighborhood(unsigned* buckets, particle_t* p, float h)
{
    /* BEGIN TASK */
    unsigned count = 0;

    int ix = static_cast<int>(p->x[0] / h);
    int iy = static_cast<int>(p->x[1] / h);
    int iz = static_cast<int>(p->x[2] / h);

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                unsigned bx = (ix + dx) & HASH_MASK;
                unsigned by = (iy + dy) & HASH_MASK;
                unsigned bz = (iz + dz) & HASH_MASK;
                
                unsigned bucket = zm_encode(bx, by, bz);
                
                bool found = false;
                for (unsigned i = 0; i < count; i++) {
                    if (buckets[i] == bucket) {
                        found = true;
                        break;
                    }
                }
                
                if (!found && count < MAX_NBR_BINS) {
                    buckets[count++] = bucket;
                }
            }
        }
    }

    return count;
    /* END TASK */
}

void hash_particles(sim_state_t* s, float h)
{
    for (int i = 0; i < HASH_SIZE; i++) {
        s->hash[i] = nullptr;
    }

    for (int i = 0; i < s->n; i++) {
        particle_t* p = &(s->part[i]);
        unsigned bucket = particle_bucket(p, h);
        p->next = s->hash[bucket];
        s->hash[bucket] = p;
    }
}
