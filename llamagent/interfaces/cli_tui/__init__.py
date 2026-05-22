"""C0 Spike: Textual TUI scaffold for LlamAgent CLI (plan v9 stage C0).

This package contains the minimal app + smoke runner used to validate
the core KPI before committing to the full TUI rewrite: alt-screen
buffer must keep host terminal scrollback accumulation under 5 KB across
100 mock turns. See docs/cli-transfer-plan.md §4 C0.
"""
