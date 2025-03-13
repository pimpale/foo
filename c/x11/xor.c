/*
 * README:
 * to compile issue
 * cc -lX11 -lm xor.c
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

#define FRAME_XSIZE 512
#define FRAME_YSIZE 512

#define INPUTONKEY(key, boolean) \
  case XK_##key: {               \
    input->key = boolean;        \
  } break;

// structs
typedef struct {
  bool w;
  bool a;
  bool s;
  bool d;
  bool q;
  bool e;
  bool mouse_down;
  int32_t mouse_x;
  int32_t mouse_y;
  int32_t previous_mouse_x;
  int32_t previous_mouse_y;
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
        // turn off keys when released
        KeySym k = XLookupKeysym(&event.xkey, 0);
        switch(k) {
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


uint32_t torgb(double val) {
  uint32_t red = rint(127+255*sin(val));
  uint32_t green = rint(127+255*sin(val + 2*M_PI/3));
  uint32_t blue = rint(127+255*sin(val + 4*M_PI/3));
  return red << 16 + green << 8 + blue;
  // actually should be the below but it looks cooler with the above
  // return (red << 16) + (green << 8) + blue;
}

double mytime = 1.0f;

void draw() {
  mytime*=1.1;
  // set to white
  if(input.q) {
    terminate = true;
  }

  XSetForeground(dis, gc, 0xFFFFFF);
  for(uint32_t y = 0; y < input.y_size; y++) {
    for(uint32_t x = 0; x < input.x_size; x++) {
      XSetForeground(dis, gc, torgb((y ^ x)/mytime));
      XDrawPoint(dis, win, gc, x, y);
    }
  }
  usleep(400000);
}

int main() {
  init_x();
  while (!terminate) {
    update_user_input(dis, &input);
    draw();
  }
  delete_x();
  return 0;
}
