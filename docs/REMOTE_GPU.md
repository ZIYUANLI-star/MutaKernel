# Remote GPU execution

Remote credentials must never be stored in this repository, command-line
arguments, result JSON, or shell history.  Use an SSH key or an ephemeral
environment variable supplied by the execution environment.

## Connection configuration

`scripts/_kgb_remote.py` reads the following variables:

| Variable | Required | Meaning |
| --- | --- | --- |
| `MUTAKERNEL_SSH_HOST` | yes | SSH host name |
| `MUTAKERNEL_SSH_USER` | yes | SSH account |
| `MUTAKERNEL_SSH_PORT` | no | SSH port; defaults to `22` |
| `MUTAKERNEL_SSH_KEY` | no | private-key path |
| `MUTAKERNEL_SSH_PASSWORD` | no | ephemeral password; prefer a key |
| `MUTAKERNEL_SSH_KNOWN_HOSTS` | no | additional known-hosts file |

The host key must already exist in the system known-hosts file or the file
provided through `MUTAKERNEL_SSH_KNOWN_HOSTS`.  Unknown host keys are rejected
instead of accepted automatically.

## Commands

```text
python scripts/_kgb_remote.py "nvidia-smi"
python scripts/_kgb_remote.py --put LOCAL_FILE REMOTE_PATH
python scripts/_kgb_remote.py --get REMOTE_PATH LOCAL_FILE
python scripts/_kgb_remote.py --script LOCAL_SCRIPT [TIMEOUT_SECONDS]
```

Do not embed secrets in helper scripts.  Experiment outputs must record only a
non-sensitive machine identifier, GPU model, driver/CUDA versions, repository
commit, dataset hash, and experiment manifest.

## Incident response

If a credential is committed:

1. revoke or rotate it immediately;
2. make the repository private while investigating, if possible;
3. remove the credential from the current tree;
4. purge it from Git history with repository-owner approval;
5. invalidate old clones and credentials;
6. run a full secret scan before making the repository public again.
