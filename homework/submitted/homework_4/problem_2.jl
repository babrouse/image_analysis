# Image Analysis Homework 4 - Problem 2
# Bret A. Brouse Jr. - 10.23.2024

# Preload packages
using Images;
using Plots;
using StatsBase;

# Functions
function int_check(img, row, col)
    row = (row * 33) - 1
    col = (col * 33) - 1

    println(img[row, col], " | ", img[row, col+1], " | ", img[row, col+2])
    println(img[row+1, col], " | ", img[row+1, col+1], " | ", img[row+1, col+2])
    println(img[row+2, col], " | ", img[row+2, col+1], " | ", img[row+2, col+2])
end;

function extract_emit(img)
    vec = []

    for i = 33:33:size(img)[1]
        for j = 33:33:size(img)[2]
            push!(vec, img[i, j])
        end
    end

    return vec
end;

function count_emitters(vec)
    mono, dim, tri = [], [], []

    for i = 1:1:length(vec)
        if vec[i] ≤ 0.2
            push!(mono, vec[i])
        elseif 0.2 < vec[i] ≤ 0.35
            push!(dim, vec[i])
        elseif 0.35 < vec[i] ≤ 1.0
            push!(tri, vec[i])
        end
    end

    return mono, dim, tri
end;

# Load up the images and convert to float32
emit_orig = load("images/emitters_33px_100ph.png")
emit_nons = load("images/emitters_33px_1000ph_noNoise.png")

emit_orig = Float32.(emit_orig)
emit_nons = Float32.(emit_nons)

#= I think the plan is to define some squares, perhaps the 3x3 grid around each center point and 
perform 3 different thresholds for each emitter? I'm not sure but based on visual inspection it 
looks like we have three different intensities. I'd guess there are 11 trimers, 20 dimers, and 69
monomers. First, I'm going to see if I can classify the intensities at the centers of these manually
then I'll try to thresh or blur based on those. =#

#= After visually inspecting some points on the non-noisy image, I decided to check if the emitters
positioned at (1,1) and (2, 1) are the same. I also compared (1, 2) with (3, 2) and (2, 6) with (4, 4)
as it appears these should be monomers, dimers, and trimers respectively. =#

# Checking to make sure these look similar
int_check(emit_nons, 1, 1); int_check(emit_nons, 2, 1)
int_check(emit_nons, 1, 2); int_check(emit_nons, 3, 2)
int_check(emit_nons, 2, 6); int_check(emit_nons, 4, 4)

# intensities = vec(emit_nons)
# non_zero_intensities = filter(x -> x > 0.05, intensities)

# Switching gears let's extract just the emitter centers in the non-noise version
# no_noise_histo = histogram(emit_vec, bins=50, color=:purple)

mon, dim, tri = count_emitters(no_noise_vec)
println(length(mon), ",", length(dim), ",", length(tri)) 
# This confirms the values of 69, 20, and 11 for monomers, dimers, and trimers respectively


no_noise_vec = extract_emit(emit_nons)
noise_vec = extract_emit(emit_orig)

no_noise_histo = histogram(no_noise_vec, 
                           title="non-noise histo", 
                           xlabel="intensity", 
                           ylabel="frequency", 
                           legend=false, 
                           bins=50, 
                           color=:purple)

noise_histo = histogram(noise_vec, 
                        title="non-noise histo", 
                        xlabel="intensity", 
                        ylabel="frequency", 
                        legend=false, 
                        bins=50, 
                        color=:orange)

# I think now I'm going to try what we did from homework 2 to try and get rid of the background 'noise'
