#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#include "vec3.hpp"
#include "state.hpp"

static void reflect_bc(sim_state_t* s);

void leapfrog_step(sim_state_t* s, double dt)
{
    int n = s->n;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        particle_t* p = s->part + i;
        vec3_saxpy(p->vh, dt,  p->a);
        vec3_copy(p->v, p->vh);
        vec3_saxpy(p->v, dt/2, p->a);
        vec3_saxpy(p->x, dt,   p->vh);
    }
    reflect_bc(s);
}

void leapfrog_start(sim_state_t* s, double dt)
{
    int n = s->n;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        particle_t* p = s->part + i;
        vec3_copy(p->vh, p->v);
        vec3_saxpy(p->vh, dt/2, p->a);
        vec3_saxpy(p->v,  dt,   p->a);
        vec3_saxpy(p->x,  dt,   p->vh);
    }
    reflect_bc(s);
}

static void damp_reflect(int which, float barrier, 
                         float* x, float* v, float* vh)
{
    const float DAMP = 0.75;

    if (v[which] == 0)
        return;

    float tbounce = (x[which]-barrier)/v[which];
    vec3_saxpy(x, -(1-DAMP)*tbounce, v);

    x[which]  = 2*barrier-x[which];
    v[which]  = -v[which];
    vh[which] = -vh[which];

    vec3_scalev(v,  DAMP);
    vec3_scalev(vh, DAMP);
}

static void reflect_bc(sim_state_t* s)
{
    const float XMIN = 0.0;
    const float XMAX = 1.0;
    const float YMIN = 0.0;
    const float YMAX = 1.0;
    const float ZMIN = 0.0;
    const float ZMAX = 1.0;

    int n = s->n;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        float* vh = s->part[i].vh;
        float* v  = s->part[i].v;
        float* x  = s->part[i].x;
        if (x[0] < XMIN) damp_reflect(0, XMIN, x, v, vh);
        if (x[0] > XMAX) damp_reflect(0, XMAX, x, v, vh);
        if (x[1] < YMIN) damp_reflect(1, YMIN, x, v, vh);
        if (x[1] > YMAX) damp_reflect(1, YMAX, x, v, vh);
        if (x[2] < ZMIN) damp_reflect(2, ZMIN, x, v, vh);
        if (x[2] > ZMAX) damp_reflect(2, ZMAX, x, v, vh);
    }
}