/*
 * README:
 * to compile issue
 * cc -lOpenCL -lX11 -lm main.c
 * to run, do
 * ./a.out
 * run
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "CL/opencl.h"

#include <X11/Xlib.h>
#include <X11/Xos.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#define INITIAL_FRAME_XSIZE 500
#define INITIAL_FRAME_YSIZE 500

// structs
typedef struct {
  bool w;
  bool a;
  bool s;
  bool d;
  bool q;
  bool e;
  bool mouse_down;
  uint32_t mouse_x;
  uint32_t mouse_y;
  uint32_t previous_mouse_x;
  uint32_t previous_mouse_y;
  uint32_t x_size;
  uint32_t y_size;
} UserInput;

// Program control variables
bool terminate = false;
UserInput user_input = {0};

// here are our X variables
Display *dis;
int screen;
Window win;
GC gc;

// and our OpenCL ones
cl_context context;
cl_command_queue queue;
cl_device_id device;

// kernel sources

const char *raygen_source =
    "float3 quat_mul_vec3(float3 v, float4 q) {"
    "  /* from linmath.h */"
    "  float3 t = 2 * cross(q.xyz,v);"
    "  return v + q.w * t + cross(q.xyz, t);"
    "}"
    "void kernel raygen("
    "                 const unsigned int x_size,"
    "                 const unsigned int y_size,"
    "                 const float4 quaternion,"
    "                 global float3* rays"
    "                ) {"
    "  unsigned int x = get_global_id(0);"
    "  unsigned int y = get_global_id(1);"
    "  float3 ray = (float3) { x - (x_size/2.0), y - (y_size/2.0), 300 };"
    "  ray = normalize(quat_mul_vec3(ray, quaternion));"
    "  rays[y*x_size + x] = ray;"
    "}";

const char *cast_source =
    "unsigned int float_to_color(float hue) {"
    "  unsigned int bluecomp = hue*0xFF;"
    "  unsigned int redcomp = (1-hue)*0xFF;"
    "  return bluecomp + (redcomp << 16);"
    "}"
    "float within_cube(float3 loc) {"
    "  if(loc.x > -50 && loc.x < 50 && loc.y > -50 && loc.y < 50 && loc.z > "
    "-50 && loc.z < 50) {"
    "    return (loc.z+50)/100.0;"
    "  }"
    "  return 0;"
    "}"
    "float within_sin(float3 loc) {"
    "  if(cos(loc.z/4) + cos(loc.x/4) + cos(loc.y/4) > 2) {"
    "    return (loc.z+50)/100.0;"
    "  }"
    "  return 0;"
    "}"
    "void kernel cast("
    "                 const unsigned int x_size,"
    "                 const unsigned int y_size,"
    "                 float3 eye,"
    "                 float3* rays,"
    "                 global unsigned int* framebuffer"
    "                ) {"
    "  unsigned int x = get_global_id(0);"
    "  unsigned int y = get_global_id(1);"
    "  unsigned int color = 0x000000;"
    "  float3 march_direction = normalize((float3) { x - (x_size/2.0), y - "
    "(y_size/2.0), 300 });"
    "  float3 loc = eye;"
    "  for(int i = 0; i < 200; i++) {"
    "    float val = within_sin(loc);"
    "    if(val > 0) {"
    "      color = float_to_color(val);"
    "      break;"
    "    }"
    "    loc += 0.5f * march_direction;"
    "  }"
    "  /* set array value */"
    "  framebuffer[y*x_size + x] = color;"
    "}";

void init_cl() {
  // Looking up the available GPUs
  cl_uint num = 1;
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 0, NULL, &num);
  if (num < 1) {
    fprintf(stderr, "could not find valid gpu");
    exit(1);
  }

  // grab the first gpu
  clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

  // create a compute context with the device
  context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  // create a queue
  queue = clCreateCommandQueueWithProperties(context, device, 0, NULL);
}

void init_x() {
  // open connection to x server
  dis = XOpenDisplay((char *)0);
  screen = DefaultScreen(dis);
  // create the window
  win = XCreateSimpleWindow(dis, DefaultRootWindow(dis), 0, 0,
                            INITIAL_FRAME_XSIZE, INITIAL_FRAME_YSIZE, 0, 0, 0);
  XSelectInput(dis, win,
               StructureNotifyMask | ButtonPressMask | ButtonReleaseMask |
                   PointerMotionMask | KeyPressMask | KeyReleaseMask);
  gc = XCreateGC(dis, win, 0, 0);
  XSetBackground(dis, gc, 0);
  XSetForeground(dis, gc, 0);
  XClearWindow(dis, win);
  XMapRaised(dis, win);
}

void delete_x() {
  XFreeGC(dis, gc);
  XDestroyWindow(dis, win);
  XCloseDisplay(dis);
}

#define INPUTONKEY(key, boolean) \
  case XK_##key: {               \
    input->key = boolean;        \
    break;                       \
  }
void update_user_input(Display *display, UserInput *input) {
  // get the next event and stuff it into our event variable.
  // Note:  only events we set the mask for are detected!
  int32_t previous_mouse_x = input->mouse_x;
  int32_t previous_mouse_y = input->mouse_y;

  while (XPending(dis) > 0) {
    XEvent event;
    XNextEvent(dis, &event);
    switch (event.type) {
      case ConfigureNotify: {
        XConfigureEvent xce = event.xconfigure;
        input->x_size = xce.width;
        input->y_size = xce.height;
      } break;
      case KeyPress: {
        KeySym k = XLookupKeysym(&event.xkey, 0);
        switch (k) {
          INPUTONKEY(w, true)
          INPUTONKEY(a, true)
          INPUTONKEY(s, true)
          INPUTONKEY(d, true)
          INPUTONKEY(q, true)
          INPUTONKEY(e, true)
          default: {
          } break;
        }
      } break;
      case KeyRelease: {
        KeySym k = XLookupKeysym(&event.xkey, 0);
        switch (k) {
          INPUTONKEY(w, false)
          INPUTONKEY(a, false)
          INPUTONKEY(s, false)
          INPUTONKEY(d, false)
          INPUTONKEY(q, false)
          INPUTONKEY(e, false)
          default: {
          } break;
        }
      } break;
      case ButtonPress: {
        // mouse is down
        input->mouse_down = true;
      } break;
      case ButtonRelease: {
        // mouse is up
        input->mouse_down = false;
      } break;
      case MotionNotify: {
        // set mouses
        input->mouse_x = event.xmotion.x;
        input->mouse_y = event.xmotion.y;
      } break;
      default: {
      }
    }
  }
  input->previous_mouse_x = previous_mouse_x;
  input->previous_mouse_y = previous_mouse_y;
}
#undef INPUTONKEY

// represents size of buffer in kernel
cl_uint x_size;
cl_uint y_size;
// represents location in 3d space
cl_float3 eye = {0.0, 0.0, -100.0};
// represents rotation
cl_float4 rotation = {0.0, 0.0, -100.0};

// variables for casting kernel
cl_kernel cast_kernel;
cl_mem framebuffer_cl_mem;
uint32_t *framebuffer = NULL;

cl_kernel raygen_kernel;
cl_mem rays_cl_mem;

void set_kernel_size(uint32_t x, uint32_t y) {
  size_t point_count = x * y;

  // do raygen kernel
  if (rays_cl_mem != NULL) {
    clReleaseMemObject(rays_cl_mem);
  }
  rays_cl_mem = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               point_count * sizeof(cl_float3), NULL, NULL);
  clSetKernelArg(raygen_kernel, 3, sizeof(cl_mem), &rays_cl_mem);

  // do cast kernel
  framebuffer = reallocarray(framebuffer, point_count, sizeof(uint32_t));
  if (framebuffer_cl_mem != NULL) {
    clReleaseMemObject(framebuffer_cl_mem);
  }
  framebuffer_cl_mem = clCreateBuffer(
      context, CL_MEM_READ_WRITE, point_count * sizeof(uint32_t), NULL, NULL);
  clSetKernelArg(cast_kernel, 0, sizeof(uint32_t), &x);
  clSetKernelArg(cast_kernel, 1, sizeof(uint32_t), &y);
  clSetKernelArg(cast_kernel, 3, sizeof(cl_mem), &framebuffer_cl_mem);
}

cl_kernel build_kernel(const char *source, const char *name) {
  cl_int program_status = !NULL;
  cl_program program =
      clCreateProgramWithSource(context, 1, &source, NULL, &program_status);

  if (program_status != CL_SUCCESS) {
    fprintf(stderr, "failed to create program: %d\n", program_status);
    exit(1);
  }

  // build the compute program executable
  cl_int build_status = clBuildProgram(program, 0, NULL, "-w", NULL, NULL);
  if (build_status != CL_SUCCESS) {
    fprintf(stderr, "failed to build program: %d\n", build_status);
    if (build_status == CL_BUILD_PROGRAM_FAILURE) {
      fprintf(stderr, "compilation error occured:\n");
      size_t length = !NULL;
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL,
                            &length);
      char *buffer = malloc(length);
      clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length,
                            buffer, NULL);
      fprintf(stderr, buffer);
      free(buffer);
    }
    exit(1);
  }

  // create the compute kernel
  return clCreateKernel(program, name, NULL);
}

void initialize() {
  // set up the user input
  x_size = INITIAL_FRAME_XSIZE;
  y_size = INITIAL_FRAME_YSIZE;
  user_input.x_size = INITIAL_FRAME_XSIZE;
  user_input.y_size = INITIAL_FRAME_YSIZE;
  // set up kernels
  cast_kernel = build_kernel(cast_source, "cast");
  set_kernel_size(INITIAL_FRAME_XSIZE, INITIAL_FRAME_YSIZE);
}

void loop() {
  update_user_input(dis, &user_input);
  if (user_input.x_size != x_size || user_input.y_size != y_size) {
    set_kernel_size(user_input.x_size, user_input.y_size);
    x_size = user_input.x_size;
    y_size = user_input.y_size;
  }

  if (user_input.q) {
    terminate = true;
  }

  // set eye location
  if (user_input.w) {
    eye.z += 1;
  }
  if (user_input.s) {
    eye.z += -1;
  }
  if (user_input.a) {
    eye.x += -1;
  }
  if (user_input.d) {
    eye.x += 1;
  }

  clSetKernelArg(cast_kernel, 2, sizeof(cl_float3), &eye);

  size_t point_count = x_size * y_size;

  const size_t global_work_offset[3] = {0, 0, 0};
  const size_t global_work_size[3] = {x_size, y_size, 1};
  const size_t local_work_size[3] = {1, 1, 1};

  // send kernel
  cl_int ret =
      clEnqueueNDRangeKernel(queue, cast_kernel, 2, global_work_offset,
                             global_work_size, local_work_size, 0, NULL, NULL);

  usleep(40000);

  // finish work
  clEnqueueReadBuffer(queue, framebuffer_cl_mem, CL_TRUE, 0,
                      point_count * sizeof(uint32_t), framebuffer, 0, NULL,
                      NULL);

  clFinish(queue);

  // put pixels
  for (uint32_t y = 0; y < y_size; y++) {
    for (uint32_t x = 0; x < x_size; x++) {
      XSetForeground(dis, gc, framebuffer[x_size * y + x]);
      XDrawPoint(dis, win, gc, x, y);
    }
  }
}

void finalize() {
  free(framebuffer);
  if (framebuffer_cl_mem != NULL) {
    clReleaseMemObject(framebuffer_cl_mem);
  }
  if (rays_cl_mem != NULL) {
    clReleaseMemObject(rays_cl_mem);
  }
}

int main() {
  init_x();
  init_cl();
  initialize();
  while (!terminate) {
    loop();
  }
  finalize();
  delete_x();
  return 0;
}
