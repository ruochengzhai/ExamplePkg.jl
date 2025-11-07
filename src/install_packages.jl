using Pkg

# 1. Activate local environment and set up Python for PyCall
Pkg.activate("../")
ENV["PYTHON"] = "" # Force PyCall to use its own Conda-managed Python

# 2. Add all Julia and Python packages
# Using Pkg.add() here ensures that if you add a new package 
# (like LazyArrays), it will be automatically added to your project.
Pkg.add([
    "PlutoUI", "PlutoTest", "PlutoTeachingTools", 
    "BenchmarkTools", "Distances", "StatsBase",
    "PyCall", "Conda", "LsqFit", "FITSIO", "Tables",
    "LazyArrays", "OhMyThreads", "Test"
])

Pkg.instantiate()