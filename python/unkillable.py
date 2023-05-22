#!/bin/python
import os
import sys
import time
import ctypes
import ctypes.util
import signal
import subprocess

# https://github.com/torvalds/linux/blob/v5.11/include/uapi/linux/prctl.h#L9
PR_SET_PDEATHSIG = 1

def set_pdeathsig():
    libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGUSR1) != 0:
        raise OSError(ctypes.get_errno(), 'SET_PDEATHSIG')


def shutdown():
    with open('yeeted', 'a+') as f:
        f.write('beans')


# call shutdown on sigchld
# set_pdeathsig()
# signal.signal(signal.SIGUSR1, lambda sig, frame: shutdown())
signal.signal(signal.SIGCHLD, lambda sig, frame: shutdown())

pid = os.fork()
if pid > 0:
    subprocess.run(sys.argv[1:])
    while True:
        time.sleep(1)
else:
    time.sleep(10)
