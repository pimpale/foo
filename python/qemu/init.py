#!/usr/bin/env python3
"""
init.py
--------
Utility script for bootstrapping a new qcow2 disk image and launching a QEMU
instance so that you can install an operating system from an ISO.

Typical usage:

    python init.py --iso path/to/os.iso --disk data/disk.qcow2 --disk-size 20G

The script will create the qcow2 image if it does not exist and then start a
QEMU process which boots from the ISO so that you can proceed with a standard
interactive installation. When the installation is finished, simply shut the
virtual machine down; you can then create checkpoints with qemudump.py and
restore them with qemurestore.py.
"""

import argparse
import pathlib
import subprocess
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialise a qcow2 disk image and boot QEMU with the given ISO for installation."
    )
    parser.add_argument(
        "--iso",
        required=True,
        help="Path to the ISO installation media.",
    )
    parser.add_argument(
        "--disk",
        default="data/disk.qcow2",
        help="Path to the qcow2 disk image to use/create.",
    )
    parser.add_argument(
        "--disk-size",
        default="20G",
        dest="disk_size",
        help="Size of the qcow2 image to create when it does not yet exist (e.g. 20G).",
    )
    parser.add_argument(
        "--memory",
        default="2G",
        help="Guest memory size to allocate to the VM (e.g. 2G, 4096M).",
    )
    parser.add_argument(
        "--cpus",
        default=2,
        type=int,
        help="Number of virtual CPUs to expose to the guest.",
    )
    parser.add_argument(
        "--extra",
        default="",
        help="Extra command-line arguments to append verbatim to the qemu-system-x86_64 invocation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the QEMU command that would be executed but do not actually run it.",
    )
    return parser


def create_disk_if_needed(disk_path: pathlib.Path, size: str) -> None:
    """Create *disk_path* qcow2 image of *size* if it does not yet exist."""
    if disk_path.exists():
        print(f"[init] Disk already exists: {disk_path}")
        return

    print(f"[init] Creating qcow2 disk {disk_path} of size {size} …")
    disk_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "qemu-img",
        "create",
        "-f",
        "qcow2",
        str(disk_path),
        size,
    ], check=True)


def build_qemu_command(args: argparse.Namespace) -> list[str]:
    """Construct the qemu-system-x86_64 command line according to *args*."""
    cmd: list[str] = [
        "qemu-system-x86_64",
        "-enable-kvm",
        "-m",
        args.memory,
        "-smp",
        str(args.cpus),
        "-cpu",
        "host",
        "-drive",
        f"file={args.disk},format=qcow2,if=virtio",
        "-cdrom",
        args.iso,
        "-boot",
        "d",
        "-net",
        "nic,model=virtio",
        "-net",
        "user",
        "-serial",
        "stdio",
    ]

    if args.extra:
        cmd.extend(args.extra.split())

    return cmd


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    disk_path = pathlib.Path(args.disk)
    create_disk_if_needed(disk_path, args.disk_size)

    qemu_cmd = build_qemu_command(args)

    print("[init] QEMU command:")
    print(" ".join(qemu_cmd))

    if args.dry_run:
        print("[init] --dry-run specified; exiting without launching QEMU.")
        return

    # Hand off control to QEMU. The call is blocking so that Ctrl-C in this
    # script propagates to the child process and terminates the VM as well.
    try:
        subprocess.run(qemu_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)


if __name__ == "__main__":  # pragma: no cover
    main() 