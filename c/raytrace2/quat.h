#ifndef 

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
