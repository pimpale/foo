#!/usr/bin/env python3
"""
qemurestore.py
--------------
Launch a new QEMU/KVM virtual machine and restore it to a previously saved
snapshot produced by *qemudump.py*.

The snapshot file itself is *read-only* to QEMU, so you can reuse it any number
of times. Use the *--readonly-dump* flag if you prefer the script to make a
temporary copy instead (useful on certain file-systems where QEMU may attempt
an in-place update).
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
import tempfile


# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Spawn a VM and restore it from a qemudump snapshot.")
    parser.add_argument(
        "--dump",
        default="data/vm_state.dump",
        help="Path to the snapshot produced by qemudump.py",
    )
    parser.add_argument(
        "--disk",
        default="data/disk.qcow2",
        help="Path to the guest qcow2 disk image that belongs to the snapshot.",
    )
    parser.add_argument(
        "--memory",
        default="2G",
        help="RAM to allocate to the guest (e.g. 2G, 4096M)",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=2,
        help="Number of virtual CPUs",
    )
    parser.add_argument(
        "--extra",
        default="",
        help="Extra arguments to append verbatim to the qemu-system-x86_64 command line.",
    )
    parser.add_argument(
        "--readonly-dump",
        action="store_true",
        help="Copy the snapshot to a temporary file before restoring so that the original remains untouched.",
    )
    return parser


# -----------------------------------------------------------------------------


def maybe_copy_dump(src: pathlib.Path) -> pathlib.Path:
    """Return *src* or a temporary copy if requested."""
    tmp_dir = tempfile.mkdtemp(prefix="qemu_restore_")
    dst = pathlib.Path(tmp_dir) / src.name
    shutil.copy2(src, dst)
    return dst


# -----------------------------------------------------------------------------


def main() -> None:
    args = build_arg_parser().parse_args()

    dump_path = pathlib.Path(args.dump).expanduser()
    if not dump_path.is_file():
        sys.exit(f"[restore] Snapshot not found: {dump_path}")

    disk_path = pathlib.Path(args.disk).expanduser()
    if not disk_path.is_file():
        sys.exit(f"[restore] Disk image not found: {disk_path}")

    if args.readonly_dump:
        print("[restore] --readonly-dump enabled; copying snapshot to a temporary file …")
        dump_path = maybe_copy_dump(dump_path)

    qemu_cmd: list[str] = [
        "qemu-system-x86_64",
        "-enable-kvm",
        "-m",
        args.memory,
        "-smp",
        str(args.cpus),
        "-cpu",
        "host",
        "-drive",
        f"file={disk_path},format=qcow2,if=virtio",
        "-incoming",
        f"file:{dump_path}",
        "-nic",
        "user,model=virtio",
        "-serial",
        "stdio",
    ]

    if args.extra:
        qemu_cmd.extend(args.extra.split())

    print("[restore] Launching VM with command:")
    print(" ".join(qemu_cmd))

    try:
        subprocess.run(qemu_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)


if __name__ == "__main__":
    main()
