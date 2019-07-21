from tkinter import *
from time import sleep
from math import *

master = Tk()

canvas_width = 500
canvas_height = 800
w = Canvas(master, 
           width=canvas_width,
           height=canvas_height)
w.pack()

blockNum = 12
blockX = [ 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400]
blockY = [ 100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 300, 300]
blockHealth = [4,4,4,4,4,4,4,4,4,4,4,4] 

# X and Y position of ball
ballx = canvas_width/2;
bally = 650;

# X and Y velocity of ball
ballxv = 1;
ballyv = 1;

# X and Y position of paddle
paddlex = canvas_width/2
paddley = 700


def drawBlock(health, x, y): 
    # check if box is not dead
    if health != 0:
        # Decide what color the actual block should be
        if health == 1:
            color = '#FF0000'
        elif health == 2:
            color = '#FF8800'
        elif health == 3:
            color = '#FFFF00'
        elif health == 4:
            color = '#00FF00'
        # what the onscreen location of the box should be
        w.create_rectangle(x-40, y-40, x+40, y+40, fill=color)

def getScore():
    if bally > 800:
        return 'Game over'
    else:
        healthTaken = 0
        for i in range(0, blockNum):
            healthTaken += 4 - blockHealth[i]
        if healthTaken != 4*blockNum:
            return 'Score: ' + str(healthTaken)
        else:
            return 'Game won'

def drawGUI():

    # Draw ball
    w.create_oval(ballx-10, bally-10, ballx+10,bally+10, fill='#FF00FF')

    # Draw paddle
    w.create_rectangle(paddlex-40, paddley-5, paddlex+40, paddley+5, fill='#FF00AA')

    # Draw blocks
    for i in range(0, blockNum):
        drawBlock(blockHealth[i], blockX[i], blockY[i])

    # Draw score
    w.create_text(canvas_width/2, 750, text=getScore())



# Takes X and Y and returns -1 or the box number
def touchingVertBlocks(x, y):
    # check top for intersection
    for i in range(0, blockNum):
        if blockHealth[i] > 0 and abs(bally - (blockY[i] - 40)) < 2 and abs(ballx - blockX[i]) < 40:
            return i
    # check bottom for intersection
    for i in range(0, blockNum):
        if blockHealth[i] > 0 and abs(bally - (blockY[i] + 40)) < 2 and abs(ballx - blockX[i]) < 40:
            return i
    # Otherwise return -1
    return -1


# Takes X and Y and returns -1 or the box number
def touchingHorzBlocks(x, y):
    # check top for intersection
    for i in range(0, blockNum):
        if blockHealth[i] > 0 and abs(ballx - (blockX[i] - 40)) < 2 and abs(bally - blockY[i]) < 40:
            return i
    # check bottom for intersection
    for i in range(0, blockNum):
        if blockHealth[i] > 0 and abs(ballx - (blockX[i] + 40)) < 2 and abs(bally - blockY[i]) < 40:
            return i
    return -1



def updateBall():
    global ballx
    global bally
    global ballxv
    global ballyv
    ballx += ballxv
    bally += ballyv
    
    # bounce off walls
    if ballx > canvas_width or ballx < 0:
        ballxv = -ballxv

    # bounce off ceiling
    if bally < 0:
        ballyv = -ballyv

    # bounce off paddle
    if abs(bally - paddley) < 2 and abs(ballx - paddlex) < 40:
        ballyv = -ballyv

    # touching the top or bottom surfaces
    vertBlockIndex = touchingVertBlocks(ballx, bally)
    if vertBlockIndex != -1:
        blockHealth[vertBlockIndex] -= 1
        ballyv = -ballyv

    # touchcing the left or right surfaces
    horzBlockIndex = touchingHorzBlocks(ballx, bally)
    if horzBlockIndex != -1:
        blockHealth[horzBlockIndex] -= 1
        ballxv = -ballxv



def updatePaddle():
    global paddlex
    x = w.winfo_pointerx() - w.winfo_rootx()
    if x > paddlex and paddlex < canvas_width -40:
        paddlex += 1
    elif paddlex > 40:
        paddlex -= 1

    
while True:
    drawGUI()
    updateBall()
    updatePaddle()
    w.update()
    sleep(0.005)
    w.delete("all")
    
