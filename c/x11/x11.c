/*
 * README:
 * to compile issue
 * cc -lX11 x11.c
 * to run, do
 * ./a.out
 */

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

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
} UserInput;

// Program control variables
bool terminate = false;
UserInput input = {0};

// here are our X variables
Display *dis;
int screen;
Window win;
GC gc;

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

void update_user_input(Display *dis, UserInput* input) {
  // get the next event and stuff it into our event variable.
  // Note:  only events we set the mask for are detected!
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
        switch(k) {
          case XK_a: {
            input->a = true;
          } break;
          case XK_w: {
            input->w = true;
          } break;
          case XK_s: {
            input->s = true;
          } break;
          case XK_d: {
            input->d = true;
          } break;
          case XK_q: {
            input->q = true;
          } break;
          case XK_e: {
            input->e = true;
          } break;
          default: {
          } break;
        }
      } break;
      case KeyRelease: {
        KeySym k = XLookupKeysym(&event.xkey, 0);
        switch(k) {
          case XK_a: {
            input->a = false;
          } break;
          case XK_w: {
            input->w = false;
          } break;
          case XK_s: {
            input->s = false;
          } break;
          case XK_d: {
            input->d = false;
          } break;
          case XK_q: {
            input->q = false;
          } break;
          case XK_e: {
            input->e = false;
          } break;
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
        input->previous_mouse_x = input->mouse_x;
        input->previous_mouse_y = input->mouse_y;
        input->mouse_x = event.xmotion.x;
        input->mouse_y = event.xmotion.y;
      } break;
      default: {
      }
    }
  }
}

void draw() {
  // blank screen
  XSetForeground(dis, gc, 0x000000);
  XFillRectangle(dis,win,gc,0,0,input.x_size,input.y_size);

  // put ellipse
  XSetForeground(dis, gc, 0xFF0000);
  for(int i = 0; i < 100; i++) {
    XFillArc(dis,win,gc,i*2, i*10, 10,10, 0, 360*64);
  }
}

int main() {
  init_x();
  while (!terminate) {
    update_user_input(dis, &input);
    draw();
    usleep(100000);
  }
  delete_x();
  return 0;
}
