"""Helper: run a command on a remote GPU server over SSH.

Usage:
    python scripts/_kgb_remote.py "command to run"
    python scripts/_kgb_remote.py --put localfile remotepath
    python scripts/_kgb_remote.py --get remotepath localfile

Connection parameters are read from the environment.  No credentials should
ever be committed to the repository.

Required environment variables:

    MUTAKERNEL_SSH_HOST
    MUTAKERNEL_SSH_USER

Optional environment variables:

    MUTAKERNEL_SSH_PORT          (default: 22)
    MUTAKERNEL_SSH_KEY           (private-key path)
    MUTAKERNEL_SSH_PASSWORD      (prefer a key; never persist this value)
    MUTAKERNEL_SSH_KNOWN_HOSTS   (additional known-hosts file)
"""
from __future__ import annotations

import os
import shlex
import sys
import uuid

import paramiko


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is not set: {name}")
    return value


def _client() -> paramiko.SSHClient:
    host = _required_env("MUTAKERNEL_SSH_HOST")
    user = _required_env("MUTAKERNEL_SSH_USER")
    port = int(os.environ.get("MUTAKERNEL_SSH_PORT", "22"))
    key_filename = os.environ.get("MUTAKERNEL_SSH_KEY") or None
    password = os.environ.get("MUTAKERNEL_SSH_PASSWORD") or None

    c = paramiko.SSHClient()
    c.load_system_host_keys()
    known_hosts = os.environ.get("MUTAKERNEL_SSH_KNOWN_HOSTS")
    if known_hosts:
        c.load_host_keys(os.path.expanduser(known_hosts))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect(
        host,
        port=port,
        username=user,
        password=password,
        key_filename=os.path.expanduser(key_filename) if key_filename else None,
        timeout=30, banner_timeout=30, auth_timeout=30,
        look_for_keys=key_filename is None and password is None,
        allow_agent=key_filename is None and password is None,
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
    c = _client()
    remote = f"/tmp/mutakernel-{uuid.uuid4().hex}.sh"
    try:
        sftp = c.open_sftp()
        sftp.put(local_path, remote)
        sftp.close()
        stdin, stdout, stderr = c.exec_command(
            f"bash {shlex.quote(remote)}", timeout=timeout, get_pty=False)
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
        try:
            c.exec_command(f"rm -f -- {shlex.quote(remote)}", timeout=30)
        except Exception:
            pass
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
