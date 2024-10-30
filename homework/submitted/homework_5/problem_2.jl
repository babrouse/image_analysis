# Image Analysis Homework 5 - Problem 2
# Bret A. Brouse Jr. - 10.30.2024

# Preload packages
using Random, Distributions;

# Some dang functions
function intensity_1(x)
    I = 0.2 * sqrt(x)

    return I
end;

function intensity_2(x, w)
    I = (30 / (x + 20)) + (0.05 * w)

    return I
end;

function centroid_calc(I_array, xᵢ)
    A = sum(I_array .* xᵢ) / sum(I_array)

    return A
end;

# define position indices
xᵢ = 0:999;
w = rand(Poisson(10), length(xᵢ))

I₁, I₂ = [], [];
for i = 1:1:length(xᵢ)
    push!(I₁, intensity_1(xᵢ[i]))
    push!(I₂, intensity_2(xᵢ[i], w[i]))
end

centroid_calc(I₁, xᵢ)
centroid_calc(I₂, xᵢ)