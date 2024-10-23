# Image Analysis Homework 4 - Problem 3
# Bret A. Brouse Jr. - 10.23.2024

# Preload packages
using Images;
using Plots;
using SpecialFunctions;

gr()

# Functions
function euclid_dist(cent, scale, a, b)
    A = (a - cent) * scale
    B = (b - cent) * scale
    C = sqrt(A^2 + B^2)

    return C
end;

function psf(N::Int, λ::Float64, aper::Float64, scale::Float64)
    # Make a grid of distances from the center of an NxN matrix
    center = (N + 1) / 2 
    r = zeros(N, N)

    for i = 1:1:N
        for j = 1:1:N
            r[i, j] = euclid_dist(center, scale, i, j)
        end
    end

    v = (2 * π / λ) * aper * r

    psf_array = zeros(N, N)

    # Now do PSF calc for each point in psf_array
    for i = 1:1:N, j = 1:1:N
        if v[i, j] == 0
            psf_array[i, j] = 1 # special case for v = 0
        else
            psf_array[i, j] = 4 * (besselj1(v[i, j]) / v[i, j])^2 # given point spread function
        end
    end

    return psf_array
end;

# variables to tweak
N = 101;
λ🟢 = 0.5;
λ🔵 = 0.4;
aper₁ = 0.9;
aper₂ = 0.5;
scale = 0.01;

psf₁ = psf(N, λ🟢, aper₁, scale)
psf₂ = psf(N, λ🔵, aper₁, scale)
psf₃ = psf(N, λ🔵, aper₂, scale)

heatmap(psf₁, title="PSF with λ=0.5μm, NA=0.9")
heatmap(psf₂, title="PSF with λ=0.4μm, NA=0.9")
heatmap(psf₃, title="PSF with λ=0.4μm, NA=0.5")