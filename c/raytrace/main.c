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

#include "linmath.h"

#define FRAME_XSIZE 500
#define FRAME_YSIZE 500
#define SCALE 5

// structs
typedef struct {
  bool w;
  bool a;
  bool s;
  bool d;
  bool Left;
  bool Right;
  bool Up;
  bool Down;
  bool q;
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
  win = XCreateSimpleWindow(dis, DefaultRootWindow(dis), 0, 0, FRAME_XSIZE,
                            FRAME_YSIZE, 0, 0, 0);
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
          INPUTONKEY(Left, true)
          INPUTONKEY(Right, true)
          INPUTONKEY(Up, true)
          INPUTONKEY(Down, true)
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
          INPUTONKEY(Left, false)
          INPUTONKEY(Right, false)
          INPUTONKEY(Up, false)
          INPUTONKEY(Down, false)
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

uint32_t *framebuffer = NULL;
cl_mem framebuffer_cl_mem;
cl_kernel kernel;

// represents the eye position
cl_float3 eye = {0.0, 0.0, -100.0};
// represents rotation
cl_float4 rotation = {
    0.0,
    0.0,
    0.0,
    1.0,
};

// time since start
cl_float current_time = 0.0f;

void set_eye(cl_float3 new_eye) {
  eye = new_eye;
  clSetKernelArg(kernel, 3, sizeof(cl_float3), &new_eye);
}

void set_rotation(cl_float4 new_rotation) {
  // set globals
  rotation = new_rotation;
  clSetKernelArg(kernel, 4, sizeof(cl_float4), &new_rotation);
}

void set_size(uint32_t x, uint32_t y) {
  x_size = x;
  y_size = y;
  size_t point_count = x * y;
  framebuffer = reallocarray(framebuffer, point_count, sizeof(uint32_t));
  if (framebuffer_cl_mem != NULL) {
    clReleaseMemObject(framebuffer_cl_mem);
  }
  cl_int ret = !NULL;
  framebuffer_cl_mem = clCreateBuffer(
      context, CL_MEM_READ_WRITE, point_count * sizeof(uint32_t), NULL, NULL);
  clSetKernelArg(kernel, 0, sizeof(uint32_t), &x);
  clSetKernelArg(kernel, 1, sizeof(uint32_t), &y);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &framebuffer_cl_mem);
}

void initialize() {
  const char *kernel_source =
      "float3 quat_mul_vec3(float3 v, float4 q) {"
      "  /* from linmath.h */"
      "  float3 t = 2 * cross(q.xyz,v);"
      "  return v + q.w * t + cross(q.xyz, t);"
      "}"
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
      "float within_sphere(float3 loc) {"
      "  if(distance(loc, (float3) {0,0,0}) < 50) {"
      "    return (loc.z+50)/100.0;"
      "  }"
      "  return 0;"
      "}"
      "float within_sin(float3 loc) {"
      "  return cos(loc.z/4) + cos(loc.x/5) + cos(loc.y/5) - 2;"
      "}"
      "void kernel cast("
      "                 const unsigned int x_size,"
      "                 const unsigned int y_size,"
      "                 global unsigned int* framebuffer,"
      "                 const float3 eye,"
      "                 const float4 quat,"
      "                 const float time"
      "                ) {"
      "  unsigned int x = get_global_id(0);"
      "  unsigned int y = get_global_id(1);"
      "  unsigned int color = 0x000000;"
      "  float3 march_direction = (float3) {"
      "       x - (x_size/2.0),"
      "       y - (y_size/2.0),"
      "       y_size"
      "     };"
      "  march_direction = 0.2f*normalize(quat_mul_vec3(march_direction, quat));"
      "  float3 loc = eye;"
      "  for(int i = 0; i < 200; i++) {"
      "    float val = within_sin(loc);"
      "    if(val > 0) {"
      "      color = float_to_color(val);"
      "      break;"
      "    }"
      "    loc += march_direction;"
      "  }"
      "  /* set array value */"
      "  framebuffer[y*x_size + x] = color;"
      "}";

  cl_program program =
      clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);

  // build the compute program executable
  cl_int ret = clBuildProgram(program, 0, NULL, "-w", NULL, NULL);
  if (ret != CL_SUCCESS) {
    fprintf(stderr, "failed to build program: %d\n", ret);
    if (ret == CL_BUILD_PROGRAM_FAILURE) {
      fprintf(stderr, "compilation error\n");
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
  kernel = clCreateKernel(program, "cast", NULL);
  set_size(FRAME_XSIZE, FRAME_YSIZE);
  user_input.x_size = FRAME_XSIZE;
  user_input.y_size = FRAME_YSIZE;
}

void loop() {
  update_user_input(dis, &user_input);

  if (user_input.x_size/SCALE != x_size/SCALE || user_input.y_size != y_size) {
    set_size(user_input.x_size/SCALE, user_input.y_size/SCALE);
  }

  if (user_input.q) {
    terminate = true;
  }

  // move value
  const float mval = 5;
  // rotate value
  const float rval = 0.05;

  // set rotation
  cl_float3 horizontal_axis = {1.0f, 0.0f, 0.0f};
  cl_float3 vertical_axis = {0.0f, 1.0f, 0.0f};
  cl_float3 depth_axis = {0.0f, 0.0f, 1.0f};

  cl_float4 new_rotation = rotation;
  if (user_input.Left) {
    quat q;
    quat_rotate(q, -rval, vertical_axis.s);
    cl_float4 current_rotation = new_rotation;
    quat_mul(new_rotation.s, current_rotation.s, q);
  }
  if (user_input.Right) {
    quat q;
    quat_rotate(q, rval, vertical_axis.s);
    cl_float4 current_rotation = new_rotation;
    quat_mul(new_rotation.s, current_rotation.s, q);
  }
  if (user_input.Up) {
    quat q;
    quat_rotate(q, rval, horizontal_axis.s);
    cl_float4 current_rotation = new_rotation;
    quat_mul(new_rotation.s, current_rotation.s, q);
  }
  if (user_input.Down) {
    quat q;
    quat_rotate(q, -rval, horizontal_axis.s);
    cl_float4 current_rotation = new_rotation;
    quat_mul(new_rotation.s, current_rotation.s, q);
  }

  set_rotation(new_rotation);

  cl_float3 rotated_horizontal_axis;
  cl_float3 rotated_vertical_axis;
  cl_float3 rotated_depth_axis;

  quat_mul_vec3(rotated_horizontal_axis.s, rotation.s, horizontal_axis.s);
  quat_mul_vec3(rotated_vertical_axis.s, rotation.s, vertical_axis.s);
  quat_mul_vec3(rotated_depth_axis.s, rotation.s, depth_axis.s);

  // set eye location
  cl_float3 new_eye = eye;
  if (user_input.w) {
    vec3_add(new_eye.s, new_eye.s, rotated_depth_axis.s);
  }
  if (user_input.s) {
    vec3_sub(new_eye.s, new_eye.s, rotated_depth_axis.s);
  }
  if (user_input.a) {
    vec3_sub(new_eye.s, new_eye.s, rotated_horizontal_axis.s);
  }
  if (user_input.d) {
    vec3_add(new_eye.s, new_eye.s, rotated_horizontal_axis.s);
  }
  set_eye(new_eye);

  current_time++;
  clSetKernelArg(kernel, 5, sizeof(cl_float), &current_time);

  size_t point_count = x_size * y_size;

  const size_t global_work_offset[3] = {0, 0, 0};
  const size_t global_work_size[3] = {x_size, y_size, 1};
  const size_t local_work_size[3] = {1, 1, 1};

  // send kernel
  cl_int ret =
      clEnqueueNDRangeKernel(queue, kernel, 2, global_work_offset,
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
      XFillRectangle(dis, win, gc, x*SCALE, y*SCALE, SCALE, SCALE);
    }
  }
}

void finalize() {
  free(framebuffer);
  if (framebuffer_cl_mem != NULL) {
    clReleaseMemObject(framebuffer_cl_mem);
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
