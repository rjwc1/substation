import dace
import numpy as np

@dace.program
def my_function(A: dace.float64[10000]):
    return A + 1

A = np.random.rand(10000)

with dace.profile(repetitions=100, warmup=10) as prof:  # Enable profiling
  my_function(A)

# Optionally, the following code will print each individual time of the first call
sdfg, timing = prof.times[0]
print(timing)