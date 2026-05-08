"""v3.8.2 A1 deprecation shim — relocated to ``llamagent.core.snapshot``.

Pre-v3.8.2 ``SnapshotService`` lived here. The class moved to the core
layer to break the ``core/agent.py → modules/tools/snapshot.py`` reverse
import (P5 violation). This shim keeps the old import path working for
two release cycles (v3.8.2 + v3.9) so third-party code that imported
``llamagent.modules.tools.snapshot:SnapshotService`` keeps loading,
with a DeprecationWarning to flag the migration.

Slated for deletion in v3.10.
"""

import warnings

from llamagent.core.snapshot import SnapshotConfig, SnapshotService  # re-export

warnings.warn(
    "llamagent.modules.tools.snapshot has moved to llamagent.core.snapshot. "
    "Update your import; the shim will be removed in v3.10.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SnapshotService", "SnapshotConfig"]
