"""C0 Spike KPI checker.

Parses a `script(1)` typescript log to isolate what actually contributes
to the HOST terminal scrollback after a Textual TUI session exits.

Background (plan v10 §4 C0 KPI revision):
- `wc -c <log>` was the v9 KPI but is misleading — it measures the full
  PTY byte stream including transient alt-screen frame redraws (typically
  1 MB+ for 100 turns), not what lands in the host's main scrollback.
- The accurate KPI is "bytes emitted OUTSIDE the alt-screen bracket"
  (before enter `\\x1b[?1049h` and after exit `\\x1b[?1049l`), with
  ANSI control sequences stripped to find any *visible* leak.

Usage:
    python -m llamagent.interfaces.cli_tui.smoke_kpi_check /tmp/tui_smoke.log

Exits non-zero if visible-after-exit bytes exceed threshold (default 100).
"""
import argparse
import re
import sys


ENTER_ALTSCREEN = b"\x1b[?1049h"
EXIT_ALTSCREEN = b"\x1b[?1049l"

# CSI sequences: ESC [ <params> <intermediate> <final>
_CSI = re.compile(rb"\x1b\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]")
# Charset / 2-char escapes: ESC <(/)/+/=/>/etc> <optional letter>
_SHORT = re.compile(rb"\x1b[<>=()][0-9A-Za-z]?")
# OSC sequences: ESC ] ... BEL or ESC \
_OSC = re.compile(rb"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


def strip_ansi(data: bytes) -> bytes:
    data = _CSI.sub(b"", data)
    data = _OSC.sub(b"", data)
    data = _SHORT.sub(b"", data)
    return data


def analyze(log_path: str, threshold: int = 100) -> int:
    with open(log_path, "rb") as f:
        data = f.read()

    total = len(data)
    enter = data.find(ENTER_ALTSCREEN)
    exit_pos = data.find(EXIT_ALTSCREEN)

    print(f"Log file: {log_path}")
    print(f"Total capture: {total:,} bytes ({total / 1024:.1f} KB)")

    if enter < 0 or exit_pos < 0 or exit_pos <= enter:
        print("FAIL: could not locate alt-screen brackets — TUI may not have run", file=sys.stderr)
        return 2

    before = data[:enter]
    inside = data[enter:exit_pos + len(EXIT_ALTSCREEN)]
    after = data[exit_pos + len(EXIT_ALTSCREEN):]

    print(f"Before alt-screen enter (byte 0..{enter}): {len(before):,} bytes")
    print(f"Inside  alt-screen      (transient)      : {len(inside):,} bytes ({len(inside) / 1024:.1f} KB)")
    print(f"After   alt-screen exit                  : {len(after):,} bytes")

    visible_after = strip_ansi(after).rstrip(b"\r\n\t ")
    visible_before = strip_ansi(before).rstrip(b"\r\n\t ")
    scrollback_visible = len(visible_before) + len(visible_after)

    print()
    print(f"Visible (stripped) bytes BEFORE alt-screen: {len(visible_before)}")
    print(f"Visible (stripped) bytes AFTER  alt-screen: {len(visible_after)}")
    print(f"Total visible scrollback contribution     : {scrollback_visible} bytes")

    if visible_after:
        print()
        print("=== Visible after-exit content (first 500 bytes) ===")
        print(repr(visible_after[:500]))

    print()
    if scrollback_visible <= threshold:
        print(f"PASS: scrollback visible {scrollback_visible} <= threshold {threshold}")
        return 0
    else:
        print(f"FAIL: scrollback visible {scrollback_visible} > threshold {threshold}", file=sys.stderr)
        return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log_path", help="Path to script(1) typescript log")
    p.add_argument("--threshold", type=int, default=100,
                   help="Max visible (post-strip) bytes outside alt-screen (default: 100)")
    args = p.parse_args()
    return analyze(args.log_path, args.threshold)


if __name__ == "__main__":
    sys.exit(main())
