# Image Analysis Homework 4 - Problem 4
# Bret A. Brouse Jr. - 10.23.2024

# Preload packages
using Images;
using Plots;
using SpecialFunctions;

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

    return psf_array / sum(psf_array)
end;

# Load the image
worm_orig = load("images/fetter_Celegans_cellfig10.jpg")

N = 101;
λ = 0.53;
aper = 0.7;
scale = 0.1;

worm_psf = psf(N, λ, aper, scale)


worm_blur = imfilter(worm_orig, Kernel.worm_psf)
display(worm_blur)