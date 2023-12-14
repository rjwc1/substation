from dace.dtypes import typeclass
import dace
import numpy as np
import math
import os
# import torch
# import torch.nn

import dace
import numpy as np

@dace.program
def twomaps(A):
    B = np.sin(A)
    return B * 2.0

a = np.random.rand(1000, 1000)
sdfg = twomaps.to_sdfg(a)
sdfg.instrument = dace.InstrumentationType.Timer  # Instrument the whole SDFG



# Instrument the individual Map scopes
for state in sdfg.nodes():
    for node in state.nodes():
        if isinstance(node, dace.nodes.MapEntry):
            node.instrument = dace.InstrumentationType.Timer


sdfg(a)

# Print the execution time in a human-readable tabular format
report = sdfg.get_latest_report()
print(report)

# The report will now contain information on each individual map. Example printout:
# Instrumentation report
# SDFG Hash: 0f02b642249b861dc94b7cbc729190d4b27cab79607b8f28c7de3946e62d5977
# ---------------------------------------------------------------------------
# Element                          Runtime (ms)
#               Min            Mean           Median         Max
# ---------------------------------------------------------------------------
# SDFG (0)
# |-State (0)
# | |-Node (0)
# | | |Map _numpy_sin__map:
# | | |          11.654         11.654         11.654         11.654
# | |-Node (5)
# | | |Map _Mult__map:
# | | |          1.524          1.524          1.524          1.524
# ---------------------------------------------------------------------------