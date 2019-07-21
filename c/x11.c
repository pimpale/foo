#include <stdio.h>
#include <stdint.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>

typedef struct {
  Display *dis;
  Window win;
  GC gc;
  uint32_t xSize;
  uint32_t ySize;
} Canvas;

void initX(Canvas *canvas, uint32_t xSize, uint32_t ySize) {
  Display *dis = XOpenDisplay((char *)0);
  Window win = XCreateSimpleWindow(dis, DefaultRootWindow(dis), 0, 0, xSize,
                                   ySize, 0, 0, 0);
  GC gc = XCreateGC(dis, win, 0, 0);
  XSetStandardProperties(dis,win,"My Window","HI!",None,NULL,0,NULL);
  XSetBackground(dis, gc, 0);
  XSetForeground(dis, gc, 0);
  XClearWindow(dis, win);
  XMapRaised(dis, win);

  canvas->dis = dis;
  canvas->win = win;
  canvas->gc = gc;
  canvas->xSize = xSize;
  canvas->ySize = ySize;
}

void freeX(Canvas *canvas) {
  XFreeGC(canvas->dis, canvas->gc);
  XDestroyWindow(canvas->dis, canvas->win);
  XCloseDisplay(canvas->dis);
}

void init_x() {
  Display *dis;
  int screen;
  Window win;
  GC gc;
	/* get the colors black and white (see section for details) */
	unsigned long black,white;

	/* use the information from the environment variable DISPLAY 
	   to create the X connection:
	*/	
	dis=XOpenDisplay((char *)0);
   	screen=DefaultScreen(dis);
	black=BlackPixel(dis,screen),	/* get color black */
	white=WhitePixel(dis, screen);  /* get color white */

	/* once the display is initialized, create the window.
	   This window will be have be 200 pixels across and 300 down.
	   It will have the foreground white and background black
	*/
   	win=XCreateSimpleWindow(dis,DefaultRootWindow(dis),0,0,	
		200, 300, 5, white, black);

	/* here is where some properties of the window can be set.
	   The third and fourth items indicate the name which appears
	   at the top of the window and the name of the minimized window
	   respectively.
	*/
	XSetStandardProperties(dis,win,"My Window","HI!",None,NULL,0,NULL);

	/* this routine determines which types of input are allowed in
	   the input.  see the appropriate section for details...
	*/
	XSelectInput(dis, win, ExposureMask|ButtonPressMask|KeyPressMask);

	/* create the Graphics Context */
        gc=XCreateGC(dis, win, 0,0);        

	/* here is another routine to set the foreground and background
	   colors _currently_ in use in the window.
	*/
	XSetBackground(dis,gc,white);
	XSetForeground(dis,gc,black);

	/* clear the window and bring it on top of the other windows */
	XClearWindow(dis, win);
	XMapRaised(dis, win);
};

int main() {

  init_x();

  for(;;);

  XInitThreads();

  Canvas c;
  initX(&c, 500, 500);

  XDrawLine(c.dis, c.win, c.gc, 50, 50, 100, 100);

  freeX(&c);
  return 0;
}
