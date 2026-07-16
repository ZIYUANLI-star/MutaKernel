"""Helper: run a command on the remote GPU server over SSH (password auth).

Usage:
    python scripts/_kgb_remote.py "command to run"
    python scripts/_kgb_remote.py --put localfile remotepath
    python scripts/_kgb_remote.py --get remotepath localfile

Connection params are for the user's own rented GPU box.
"""
from __future__ import annotations

import sys
import paramiko

HOST = "connect.nma1.seetacloud.com"
PORT = 22841
USER = "root"
PASS = "TIdIHhvPb0cT"


def _client() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        HOST, port=PORT, username=USER, password=PASS,
        timeout=30, banner_timeout=30, auth_timeout=30,
        look_for_keys=False, allow_agent=False,
    )
    return c


def run(cmd: str, timeout: int = 600) -> int:
    c = _client()
    try:
        stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout, get_pty=False)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        if out:
            sys.stdout.write(out)
        if err:
            sys.stdout.write("\n[STDERR]\n" + err)
        sys.stdout.write(f"\n[EXIT {rc}]\n")
        return rc
    finally:
        c.close()


def put(local: str, remote: str) -> int:
    c = _client()
    try:
        sftp = c.open_sftp()
        sftp.put(local, remote)
        sftp.close()
        print(f"[PUT OK] {local} -> {remote}")
        return 0
    finally:
        c.close()


def get(remote: str, local: str) -> int:
    c = _client()
    try:
        sftp = c.open_sftp()
        sftp.get(remote, local)
        sftp.close()
        print(f"[GET OK] {remote} -> {local}")
        return 0
    finally:
        c.close()


def run_script(local_path: str, timeout: int = 3600) -> int:
    """Upload a local shell script to /tmp and run it with bash."""
    import os
    c = _client()
    try:
        sftp = c.open_sftp()
        remote = "/tmp/_kgb_remote_script.sh"
        sftp.put(local_path, remote)
        sftp.close()
        stdin, stdout, stderr = c.exec_command(
            f"bash {remote}", timeout=timeout, get_pty=False)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        rc = stdout.channel.recv_exit_status()
        if out:
            sys.stdout.write(out)
        if err:
            sys.stdout.write("\n[STDERR]\n" + err)
        sys.stdout.write(f"\n[EXIT {rc}]\n")
        return rc
    finally:
        c.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("need args")
        sys.exit(2)
    if sys.argv[1] == "--put":
        sys.exit(put(sys.argv[2], sys.argv[3]))
    if sys.argv[1] == "--get":
        sys.exit(get(sys.argv[2], sys.argv[3]))
    if sys.argv[1] == "--script":
        to = int(sys.argv[3]) if len(sys.argv) > 3 else 3600
        sys.exit(run_script(sys.argv[2], to))
    sys.exit(run(sys.argv[1]))
