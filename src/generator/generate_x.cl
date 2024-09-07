#define RNG32

#define PCG6432_FLOAT_MULTI 2.3283064365386963e-10f
#define PCG6432_DOUBLE2_MULTI 2.3283064365386963e-10
#define PCG6432_DOUBLE_MULTI 5.4210108624275221700372640e-20

/**
State of pcg6432 RNG.
*/
typedef unsigned long pcg6432_state;

#define PCG6432_XORSHIFTED(s) ((uint)((((s) >> 18u) ^ (s)) >> 27u))
#define PCG6432_ROT(s) ((s) >> 59u)

#define pcg6432_macro_uint(state)                                              \
  (state = state * 6364136223846793005UL + 0xda3e39cb94b95bdbUL,               \
   (PCG6432_XORSHIFTED(state) >> PCG6432_ROT(state)) |                         \
       (PCG6432_XORSHIFTED(state) << ((-PCG6432_ROT(state)) & 31)))

/**
Generates a random 32-bit unsigned integer using pcg6432 RNG.

@param state State of the RNG to use.
*/
#define pcg6432_uint(state) _pcg6432_uint(&state)
unsigned int _pcg6432_uint(pcg6432_state *state) {
  ulong oldstate = *state;
  *state = oldstate * 6364136223846793005UL + 0xda3e39cb94b95bdbUL;
  uint xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
  uint rot = oldstate >> 59u;
  return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

/**
Seeds pcg6432 RNG.

@param state Variable, that holds state of the generator to be seeded.
@param seed Value used for seeding. Should be randomly generated for each
instance of generator (thread).
*/
void pcg6432_seed(pcg6432_state *state, unsigned long j) { *state = j; }

/**
Generates a random 64-bit unsigned integer using pcg6432 RNG.

@param state State of the RNG to use.
*/
#define pcg6432_ulong(state)                                                   \
  ((((ulong)pcg6432_uint(state)) << 32) | pcg6432_uint(state))

/**
Generates a random float using pcg6432 RNG.

@param state State of the RNG to use.
*/
#define pcg6432_float(state) (pcg6432_uint(state) * PCG6432_FLOAT_MULTI)

/**
Generates a random double using pcg6432 RNG.

@param state State of the RNG to use.
*/
#define pcg6432_double(state) (pcg6432_ulong(state) * PCG6432_DOUBLE_MULTI)

/**
Generates a random double using pcg6432 RNG. Generated using only 32 random
bits.

@param state State of the RNG to use.
*/
#define pcg6432_double2(state) (pcg6432_uint(state) * PCG6432_DOUBLE2_MULTI)

float exp_x(float lambda_x, pcg6432_state *state) {
  float x = pcg6432_float(*state);
  return log(x) / lambda_x;
}

float exp_y(float lambda_y, pcg6432_state *state) {
  float y = pcg6432_float(*state);
  bool sign = pcg6432_uint(*state) % 2;
  return sign ? -log(y) / lambda_y : log(y) / lambda_y;
}

float even_z(pcg6432_state *state) { return pcg6432_float(*state) * 2 * M_PI; }

void init_rand(pcg6432_state *state, __global long *seed) {
  int seed_index = get_global_id(0) % 7812500;
  for (int i = 0; i < 64; i++) {
    pcg6432_uint(*state);
  }
}

__kernel void generate_points(__global long *seed, __global float *buffer,
                              float lambda_x, float lambda_y) {
  pcg6432_state state;
  pcg6432_seed(&state, seed[get_global_id(0) % (500000000 / 100)]);

  for (int i = 0; i < get_global_id(0) % 100; i++) {
    pcg6432_float(state);
  }

  float p_radius = exp_x(lambda_x, &state);
  float p_z = exp_y(lambda_y, &state);
  float p_theta = even_z(&state);

  buffer[get_global_id(0) * 3] = p_radius * cos(p_theta);
  buffer[get_global_id(0) * 3 + 1] = p_radius * sin(p_theta);
  buffer[get_global_id(0) * 3 + 2] = p_z;
}