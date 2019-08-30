/*
 * README:
 * to compile issue
 * cc -lX11 -lm fibonaccispiral.c
 * to run, do
 * ./a.out
 * run
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
        // turn on keys when pressed
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
        // turn off keys when released
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

#define NUM_POINTS 1000
#define SIZE_POINTS 7
#define HIGHLIGHT_EVERY 13
double turnfraction = 1.618;
void draw() {
  // blank screen
  XSetForeground(dis, gc, 0x000000);
  XFillRectangle(dis,win,gc,0,0,input.x_size,input.y_size);

  // put ellipse
  XSetForeground(dis, gc, 0xFF0000);
  uint32_t center_x = input.x_size/2;
  uint32_t center_y = input.y_size/2;


  for(int i = 0; i < NUM_POINTS; i++) {
    double distance = 500*i / (NUM_POINTS -1.0);
    double angle = 2 * M_PI * i * turnfraction;

    uint32_t x = center_x - (SIZE_POINTS/2) + cosf(angle)*distance;
    uint32_t y = center_y - (SIZE_POINTS/2) + sinf(angle)*distance;
    if(i % HIGHLIGHT_EVERY == 0) {
      XSetForeground(dis, gc, 0xFFFF00);
      XFillArc(dis,win,gc,x, y, SIZE_POINTS, SIZE_POINTS, 0, 360*64);
      XSetForeground(dis, gc, 0xFF0000);
    } else {
      XFillArc(dis,win,gc,x, y, SIZE_POINTS, SIZE_POINTS, 0, 360*64);
    }
  }
  if(input.w) {
    turnfraction += 0.0001;
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
