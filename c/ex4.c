#!//bin/tcc -run -L/usr/include/X11 -lX11
#include <stdlib.h>
#include <stdio.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>

/* Yes, TCC can use X11 too ! */

int main(int argc, char **argv)
{
		Display *dis;
		Window win;
		GC gc;

		int x_size = 500;
		int y_size = 500;

		dis=XOpenDisplay((char *)0);
		win=XCreateSimpleWindow(dis,DefaultRootWindow(dis),0,0, x_size, y_size, 0, 0, 0);
		XSetStandardProperties(dis,win,"Now with even more confusing vectors!","4Sight",None,NULL,0,NULL);
		XSelectInput(dis, win,
						StructureNotifyMask|ButtonPressMask|ButtonReleaseMask|PointerMotionMask|KeyPressMask|KeyReleaseMask);
		gc=XCreateGC(dis, win, 0,0);
		XSetBackground(dis,gc,0);
		XSetForeground(dis,gc,0);
		XClearWindow(dis, win);
		XMapRaised(dis, win);

		while(1);

		return 0;
}
