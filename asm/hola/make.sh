#!/bin/sh
nasm -felf64 hello.asm && ld hello.o
