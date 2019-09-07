#ifndef QUAT_H
#define QUAT_H

#include <math.h>

#define LINMATH_H_DEFINE_VEC(n)                                                \
typedef float vec##n[n];                                                     \
static inline void vec##n##_add(vec##n r, vec##n const a, vec##n const b) {  \
  int i;                                                                     \
  for (i = 0; i < n; ++i) {                                                  \
    r[i] = a[i] + b[i];                                                      \
  }                                                                          \
}                                                                            \
static inline void vec##n##_sub(vec##n r, vec##n const a, vec##n const b) {  \
  int i;                                                                     \
  for (i = 0; i < n; ++i) {                                                  \
    r[i] = a[i] - b[i];                                                      \
  }                                                                          \
}                                                                            \
static inline void vec##n##_scale(vec##n r, vec##n const v, float const s) { \
  int i;                                                                     \
  for (i = 0; i < n; ++i) {                                                  \
    r[i] = v[i] * s;                                                         \
  }                                                                          \
}                                                                            \
static inline float vec##n##_mul_inner(vec##n const a, vec##n const b) {     \
  float p = 0.;                                                              \
  int i;                                                                     \
  for (i = 0; i < n; ++i) {                                                  \
    p += b[i] * a[i];                                                        \
  }                                                                          \
  return (p);                                                                \
}                                                                            \
static inline float vec##n##_len(vec##n const v) {                           \
  return (sqrtf(vec##n##_mul_inner(v, v)));                                  \
}                                                                            \
static inline void vec##n##_norm(vec##n r, vec##n const v) {                 \
  float k = 1.0f / vec##n##_len(v);                                          \
  vec##n##_scale(r, v, k);                                                   \
}                                                                            \
static inline void vec##n##_min(vec##n r, vec##n a, vec##n b) {              \
  int i;                                                                     \
  for (i = 0; i < n; ++i) {                                                  \
    r[i] = a[i] < b[i] ? a[i] : b[i];                                        \
  }                                                                          \
}                                                                            \
static inline void vec##n##_max(vec##n r, vec##n a, vec##n b) {              \
  int i;                                                                     \
  for (i = 0; i < n; ++i) {                                                  \
    r[i] = a[i] > b[i] ? a[i] : b[i];                                        \
  }                                                                          \
}

LINMATH_H_DEFINE_VEC(2)
LINMATH_H_DEFINE_VEC(3)
LINMATH_H_DEFINE_VEC(4)

static inline void vec3_mul_cross(vec3 r, vec3 const a, vec3 const b) {
  r[0] = a[1] * b[2] - a[2] * b[1];
  r[1] = a[2] * b[0] - a[0] * b[2];
  r[2] = a[0] * b[1] - a[1] * b[0];
}

static inline void vec3_reflect(vec3 r, vec3 const v, vec3 const n) {
  float p = 2.0f * vec3_mul_inner(v, n);
  int i;
  for (i = 0; i < 3; ++i) {
    r[i] = v[i] - p * n[i];
  }
}

static inline void vec4_mul_cross(vec4 r, vec4 a, vec4 b) {
  r[0] = a[1] * b[2] - a[2] * b[1];
  r[1] = a[2] * b[0] - a[0] * b[2];
  r[2] = a[0] * b[1] - a[1] * b[0];
  r[3] = 1.0f;
}

static inline void vec4_reflect(vec4 r, vec4 v, vec4 n) {
  float p = 2.0f * vec4_mul_inner(v, n);
  int i;
  for (i = 0; i < 4; ++i) {
    r[i] = v[i] - p * n[i];
  }
}



typedef float quat[4];
static inline void quat_identity(quat q) {
  q[0] = q[1] = q[2] = 0.0f;
  q[3] = 1.0f;
}
static inline void quat_add(quat r, quat a, quat b) {
  int i;
  for (i = 0; i < 4; ++i) {
    r[i] = a[i] + b[i];
  }
}
static inline void quat_sub(quat r, quat a, quat b) {
  int i;
  for (i = 0; i < 4; ++i) {
    r[i] = a[i] - b[i];
  }
}
static inline void quat_mul(quat r, quat p, quat q) {
  vec3 w;
  vec3_mul_cross(r, p, q);
  vec3_scale(w, p, q[3]);
  vec3_add(r, r, w);
  vec3_scale(w, q, p[3]);
  vec3_add(r, r, w);
  r[3] = p[3] * q[3] - vec3_mul_inner(p, q);
}
static inline void quat_scale(quat r, quat v, float s) {
  int i;
  for (i = 0; i < 4; ++i) {
    r[i] = v[i] * s;
  }
}
static inline float quat_inner_product(quat a, quat b) {
  float p = 0.0f;
  int i;
  for (i = 0; i < 4; ++i) {
    p += b[i] * a[i];
  }
  return (p);
}
static inline void quat_conj(quat r, quat q) {
  int i;
  for (i = 0; i < 3; ++i) {
    r[i] = -q[i];
  }
  r[3] = q[3];
}
static inline void quat_rotate(quat r, float angle, vec3 axis) {
  vec3 v;
  vec3_scale(v, axis, sinf(angle / 2));
  int i;
  for (i = 0; i < 3; ++i) {
    r[i] = v[i];
  }
  r[3] = cosf(angle / 2);
}
#define quat_norm vec4_norm
static inline void quat_mul_vec3(vec3 r, quat q, vec3 v) {
  /*
   * Method by Fabian 'ryg' Giessen (of Farbrausch)
  t = 2 * cross(q.xyz, v)
  v' = v + q.w * t + cross(q.xyz, t)
   */
  vec3 t;
  vec3 q_xyz = {q[0], q[1], q[2]};
  vec3 u = {q[0], q[1], q[2]};

  vec3_mul_cross(t, q_xyz, v);
  vec3_scale(t, t, 2);

  vec3_mul_cross(u, q_xyz, t);
  vec3_scale(t, t, q[3]);

  vec3_add(r, v, t);
  vec3_add(r, r, u);
}


#endif
