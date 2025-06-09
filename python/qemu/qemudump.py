#!/usr/bin/env python3
"""
qemudump.py
-----------
Save an *exact* runtime snapshot (RAM + device state) of a running QEMU/KVM
virtual machine so that it can be restored later with *qemurestore.py*.

The snapshot is obtained via QEMU's migration facilities exposed over QMP. The
VM must have been started with a QMP control socket, e.g.::

    qemu-system-x86_64 \
        -enable-kvm \
        -m 2G -smp 2 -cpu host \
        -drive file=data/disk.qcow2,format=qcow2,if=virtio \
        -qmp unix:/tmp/qmp.sock,server=on,wait=no \
        -nic user,model=virtio

Running::

    python qemudump.py --qmp-sock /tmp/qmp.sock --output data/vm_state.dump

will produce *data/vm_state.dump* containing the complete guest state. That
file can subsequently be restored any number of times with
*qemurestore.py*.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import socket
import time
from typing import Any, Dict


class QMPClient:
    """Minimal QMP helper sufficient for migration commands."""

    def __init__(self, sock_path: str | pathlib.Path):
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(str(sock_path))
        self._file = self._sock.makefile("rwb", buffering=0)

        # QMP greets first; read and discard capabilities advertisement.
        self._recv_qmp_message()  # greeting
        # Enable capabilities so we can issue further commands.
        self.cmd("qmp_capabilities")

    def _recv_qmp_message(self) -> Dict[str, Any]:
        line = self._file.readline()
        if not line:
            raise RuntimeError("Unexpected EOF from QMP socket")
        return json.loads(line.decode())

    def _send(self, msg: Dict[str, Any]) -> None:
        self._file.write(json.dumps(msg).encode() + b"\n")
        self._file.flush()

    def cmd(self, execute: str, arguments: Dict[str, Any] | None = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"execute": execute}
        if arguments:
            payload["arguments"] = arguments
        self._send(payload)

        # QMP can deliver events asynchronously; skip them until we receive the
        # actual command response (a dict with a "return" key).
        while True:
            resp = self._recv_qmp_message()
            if "return" in resp:
                return resp["return"]  # type: ignore[return-value]
            # otherwise it's an event; ignore and continue

    # ----------------------------------------------------------------------------

    def close(self) -> None:
        try:
            self._file.close()
        finally:
            self._sock.close()


# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dump QEMU VM state to a file via QMP migration.")
    parser.add_argument(
        "--qmp-sock",
        required=True,
        help="Path to the QMP UNIX socket exposed by the running VM.",
    )
    parser.add_argument(
        "--output",
        default="data/vm_state.dump",
        help="Destination file for the VM snapshot.",
    )
    parser.add_argument(
        "--poll-interval",
        default=1.0,
        type=float,
        metavar="SEC",
        help="Polling frequency (seconds) for migration completion status.",
    )
    return parser


# -----------------------------------------------------------------------------


def main() -> None:
    args = build_arg_parser().parse_args()

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[dump] Connecting to QMP at {args.qmp_sock} …")
    qmp = QMPClient(args.qmp_sock)

    uri = f"exec:cat > {out_path}"
    print(f"[dump] Initiating migration to '{uri}' …")
    qmp.cmd("migrate", {"uri": uri})

    print("[dump] Waiting for migration to complete …")
    while True:
        status: dict[str, Any] = qmp.cmd("query-migrate")  # type: ignore[assignment]
        state = status.get("status")
        if state == "completed":
            print("[dump] Snapshot completed successfully.")
            break
        if state in {"failed", "cancelled"}:
            raise RuntimeError(f"Migration failed (status = {state}): {status}")
        time.sleep(args.poll_interval)

    qmp.close()
    print(f"[dump] Snapshot stored at {out_path}")


if __name__ == "__main__":
    main()
