"""Entry point for ``python -m llamagent``.

Delegates to :func:`llamagent.main.main` so the package-level launch
works the way ``llamagent/main.py``'s own docstring promises. Without
this file ``python -m llamagent`` fails with "No module named
llamagent.__main__"; users have to type ``python -m llamagent.main``
which works but feels off-spec for a Python package.
"""
from llamagent.main import main


if __name__ == "__main__":
    main()
