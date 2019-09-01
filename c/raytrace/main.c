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

#define FRAME_XSIZE 500
#define FRAME_YSIZE 500

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
  uint32_t previous_x_size;
  uint32_t previous_y_size;
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
  int32_t previous_x_size = input->x_size;
  int32_t previous_y_size = input->y_size;
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
  input->previous_x_size = previous_x_size;
  input->previous_y_size = previous_y_size;
  input->previous_mouse_x = previous_mouse_x;
  input->previous_mouse_y = previous_mouse_y;

}
#undef INPUTONKEY

uint32_t *color_buffer = NULL;
cl_mem color_buffer_cl_mem;
cl_kernel kernel;

void set_kernel_buffer(uint32_t x, uint32_t y) {
  size_t point_count = x * y;
  color_buffer = reallocarray(color_buffer, point_count, sizeof(uint32_t));
  if (color_buffer_cl_mem != NULL) {
    clReleaseMemObject(color_buffer_cl_mem);
  }
  cl_int ret = !NULL;
  color_buffer_cl_mem =
      clCreateBuffer(context, CL_MEM_READ_WRITE, point_count*sizeof(uint32_t), NULL, NULL);
  clSetKernelArg(kernel, 0, sizeof(uint32_t), &x);
  clSetKernelArg(kernel, 1, sizeof(uint32_t), &y);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &color_buffer_cl_mem);
}

void initialize() {
  const char *kernel_source =
    "void kernel main(unsigned int x_size, unsigned int y_size, global unsigned int* color) {"
    "  unsigned int x = get_global_id(0);"
    "  unsigned int y = get_global_id(1);"
    "  color[y*x_size + x] = 63*(2.0 + sin(x/10.0) + sin(y/10.0));"
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
  kernel = clCreateKernel(program, "main", NULL);
  set_kernel_buffer(FRAME_XSIZE, FRAME_YSIZE);
  user_input.previous_x_size = FRAME_XSIZE;
  user_input.previous_y_size = FRAME_YSIZE;
  user_input.x_size = FRAME_XSIZE;
  user_input.y_size = FRAME_YSIZE;
}

void loop() {
  update_user_input(dis, &user_input);
  if (user_input.x_size != user_input.previous_x_size ||
      user_input.y_size != user_input.previous_y_size) {
    set_kernel_buffer(user_input.x_size, user_input.y_size);
  }
  uint32_t x_size = user_input.x_size;
  uint32_t y_size = user_input.y_size;

  if(user_input.q) {
    terminate = true;
  }

  size_t point_count = x_size*y_size;
  size_t buffer_size = point_count  * sizeof(uint32_t);

  // copy the info
  //clEnqueueWriteBuffer(queue, color_buffer_cl_mem, CL_TRUE, 0, buffer_size,
  //                     color_buffer, 0, NULL, NULL);

  const size_t global_work_offset[3] = {0, 0, 0};
  const size_t global_work_size[3] = {x_size, y_size, 1};
  const size_t local_work_size[3] = {1, 1, 1};

  // send kernel
  cl_int ret = clEnqueueNDRangeKernel(queue, kernel, 2, global_work_offset, global_work_size,
                         local_work_size, 0, NULL, NULL);

  usleep(10000);

  // finish work
  clEnqueueReadBuffer(queue, color_buffer_cl_mem, CL_TRUE, 0, buffer_size,
                       color_buffer, 0, NULL, NULL);

  clFinish(queue);

  // put pixels
  for (uint32_t y = 0; y < y_size; y++) {
    for (uint32_t x = 0; x < x_size; x++) {
      XSetForeground(dis, gc, color_buffer[x_size*y + x]);
      XDrawPoint(dis, win, gc, x, y);
    }
  }

}

void finalize() {
  free(color_buffer);
  if (color_buffer_cl_mem != NULL) {
    clReleaseMemObject(color_buffer_cl_mem);
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
