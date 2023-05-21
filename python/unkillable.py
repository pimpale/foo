import os
import sys
import ctypes
import ctypes.util
import signal


# https://github.com/torvalds/linux/blob/v5.11/include/uapi/linux/prctl.h#L9
PR_SET_PDEATHSIG = 1

def set_pdeathsig():
    libc = ctypes.CDLL(ctypes.util.find_library('c'), use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, signal.SIGCHLD) != 0:
        raise OSError(ctypes.get_errno(), 'SET_PDEATHSIG')


def shutdown():
    print('You pressed Ctrl+C!')
    sys.exit(0)

# call shutdown on sigchld
signal.signal(signal.SIGCHLD, lambda sig, frame: shutdown())


critical_program = sys.argv[1]
critical_program_args = sys.argv[1:]


