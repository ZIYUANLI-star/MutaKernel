"""CPU-only tests for the B11 compute-sanitizer report parser.

The parser converts compute-sanitizer text reports into per-alarm-type
buckets (blueprint Table 2: alarm types reported separately).  No GPU and no
sanitizer binary is needed: parsing is pure.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b11_compute_sanitizer import (
    classify_alarm,
    parse_sanitizer_report,
    report_status,
)

MEMCHECK_REPORT = """\
========= COMPUTE-SANITIZER
========= Invalid __global__ read of size 4 bytes
=========     at 0x1b8 in kernel(float *, float *)
=========     by thread (31,0,0) in block (0,0,0)
=========     Address 0x7f7a4e000000 is out of bounds
=========     Saved host backtrace up to driver entry point at kernel launch time
=========     Host Frame: [0x123456] in libcuda.so
========= Program hit cudaErrorLaunchFailure (error 719) due to "unspecified launch failure" on CUDA API call to cudaDeviceSynchronize.
========= ERROR SUMMARY: 2 errors
"""

RACECHECK_REPORT = """\
========= COMPUTE-SANITIZER
========= Error: Race reported between Write access at 0x50 in kernel(float *) and Read access at 0x90 in kernel(float *)
========= RACECHECK SUMMARY: 1 hazard displayed (1 error, 0 warnings)
"""

SYNCCHECK_REPORT = """\
========= COMPUTE-SANITIZER
========= Barrier error detected. Divergent thread(s) in block
=========     at 0x2c8 in kernel(float *)
========= ERROR SUMMARY: 1 error
"""

INITCHECK_REPORT = """\
========= COMPUTE-SANITIZER
========= Uninitialized __global__ memory read of size 4 bytes
=========     at 0x88 in kernel(float *)
========= ERROR SUMMARY: 1 error
"""

CLEAN_REPORT = """\
========= COMPUTE-SANITIZER
========= ERROR SUMMARY: 0 errors
"""


def test_memcheck_report_buckets():
    parsed = parse_sanitizer_report(MEMCHECK_REPORT)
    assert parsed["summary_found"] is True
    assert parsed["error_count"] == 2
    assert parsed["by_category"] == {
        "invalid_memory_access": 1,
        "api_error": 1,
    }
    assert report_status(parsed, exit_code=99, timed_out=False) == "alarms"


def test_racecheck_hazard_summary_and_bucket():
    parsed = parse_sanitizer_report(RACECHECK_REPORT)
    assert parsed["hazard_count"] == 1
    assert parsed["error_count"] is None
    assert parsed["by_category"] == {"race_hazard": 1}
    assert report_status(parsed, exit_code=99, timed_out=False) == "alarms"


def test_synccheck_barrier_bucket():
    parsed = parse_sanitizer_report(SYNCCHECK_REPORT)
    assert parsed["error_count"] == 1
    assert parsed["by_category"] == {"barrier_sync_error": 1}


def test_initcheck_uninitialized_bucket():
    parsed = parse_sanitizer_report(INITCHECK_REPORT)
    assert parsed["by_category"] == {"uninitialized_read": 1}


def test_clean_report():
    parsed = parse_sanitizer_report(CLEAN_REPORT)
    assert parsed["summary_found"] is True
    assert parsed["error_count"] == 0
    assert parsed["alarms"] == []
    assert report_status(parsed, exit_code=0, timed_out=False) == "clean"


def test_backtrace_lines_are_not_alarms():
    parsed = parse_sanitizer_report(MEMCHECK_REPORT)
    headlines = [alarm["headline"] for alarm in parsed["alarms"]]
    assert not any("Host Frame" in headline for headline in headlines)
    assert not any(headline.startswith("at 0x") for headline in headlines)


def test_status_without_summary():
    parsed = parse_sanitizer_report("no sanitizer output at all")
    assert parsed["summary_found"] is False
    assert report_status(parsed, exit_code=0, timed_out=False) == (
        "clean_no_kernel_activity")
    assert report_status(parsed, exit_code=1, timed_out=False) == "inconclusive"
    assert report_status(parsed, exit_code=None, timed_out=True) == (
        "inconclusive_timeout")


def test_classify_alarm_fallback():
    assert classify_alarm("Something entirely new") == "other"
    assert classify_alarm("Invalid __shared__ write of size 8") == (
        "invalid_memory_access")
    assert classify_alarm("Leaked 1024 bytes at 0x7f00") == "memory_leak"
