"""C0 Spike smoke runner.

Usage:
    # Normal exit smoke (alt-screen KPI):
    script -q /tmp/tui_smoke.log llamagent_env/bin/python \\
        -m llamagent.interfaces.cli_tui.smoke 100

    # Crash exit smoke (plan v9 §11 Q6 — does exception leak to scrollback?):
    script -q /tmp/tui_crash.log llamagent_env/bin/python \\
        -m llamagent.interfaces.cli_tui.smoke 5 --crash

Then run smoke_kpi_check.py to verify after-exit visible bytes ≤ threshold.

Q6 mitigation (revised after real-terminal Step-2 result FAIL):
- sys.excepthook does NOT intercept Textual-caught exceptions (Textual catches
  internally in `App._handle_exception` and uses Rich Console to print).
- Real fix: override `App._handle_exception` to log full traceback to file
  + stash one-line notice; we emit the notice to stderr AFTER `app.run()`
  returns, so it's a single ~70-byte plain line, not 2.4 KB of box-drawing.

See `LlamAgentTUISpike._handle_exception` in app.py for the implementation.
"""
import sys


def main() -> int:
    args = sys.argv[1:]
    crash_mode = "--crash" in args
    args = [a for a in args if a != "--crash"]
    n_turns = int(args[0]) if args else 10

    from llamagent.interfaces.cli_tui.app import LlamAgentTUI
    app = LlamAgentTUI(
        n_mock_turns=n_turns,
        crash_after_turns=n_turns if crash_mode else None,
    )
    app.run()
    # Plain ASCII one-liner after alt-screen exit — minimal scrollback bytes.
    notice = getattr(app, "_crash_notice", None)
    if notice:
        sys.stderr.write(notice + "\n")
    return getattr(app, "_return_code", 0) or 0


if __name__ == "__main__":
    sys.exit(main())
