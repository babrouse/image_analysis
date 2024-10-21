# Image Analysis Homework 4 - Problem 2
# Bret A. Brouse Jr. - 10.23.2024

# Preload packages
using Images;

# Load up the Images
emit_orig = load("images/emitters_33px_100ph.png")
emit_nons = load("images/emitters_33px_1000ph_noNoise.png")

# all point sources should be centered at multiples of 33x33
# so I checked to make sure at 33x33 it's the brightest point
println(emit_nons[65, 65], ", ", emit_nons[65, 66], ", ", emit_nons[65, 67])
println(emit_nons[66, 65], ", ", emit_nons[66, 66], ", ", emit_nons[66, 67])
println(emit_nons[67, 65], ", ", emit_nons[67, 66], ", ", emit_nons[67, 67])