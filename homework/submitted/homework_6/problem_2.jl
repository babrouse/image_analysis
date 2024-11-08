# Homework 6 Problem 2
# Bret A. Brouse Jr. - 11.6.2024

# Preload packages
using Random, Distributions, SpecialFunctions;
using Optim;
using Plots;

plotly()

# functions
# We want parameters x₀ and b to maximize likelihood of observing y
function mle_obj(params, x, y)
    x₀, b = params
    A = A_calc(x, x₀, b)

    # fine negative log likelihood
    log_like = sum(A .- y .* log.(A))

    return log_like
end;

function A_calc(x, x₀, b)
    A = 3 * abs.(x .- x₀).^b .+ 4
    
    return A
end

# Define the true values
x₀, b = 0.75, 2.2;

# Set up parameters
N = 20;
x = range(-3, 4, length=N);

A = A_calc(x, x₀, b); # true values A

y = []; # Need an array for y values based on A
for i = 1:1:N
    push!(y, rand(Poisson(A[i])))
end

###########################
# A lot of the above could probably be refactored
# Now to actually try and optimize

guess = [0.5, 2.0];
result = optimize(p -> mle_obj(p, x, y), guess, BFGS())
# BFGS = Broyden–Fletcher–Goldfarb–Shanno
# This algorithm uses gradient descents (I may code this by hand later)

# Run the optimizer from Optim package
x₀_est, b_est = Optim.minimizer(result)

println("Estimated x₀: ", x₀_est)
println("Estimated b: ", b_est)

# Plot simulated y data with true and estimated models
true_est_plt = plot(x, y, seriestype=:scatter, label="observed y", legend=:topright, 
                    xlabel="x", ylabel="y", title="data compared with true model and simulated model", 
                    size=(800, 600))
plot!(true_est_plt, x, A_calc(x, x₀, b), label="true model", lw=2)
plot!(true_est_plt, x, A_calc(x, x₀_est, b_est), label="estimated model", lw=2, linestyle=:dash)

# savefig(true_est_plt, "homework/submitted/homework_6/images/2c_plot.png")
# for some reason savefig doesn't work here so I screenshot it

###########################
# Problem 2d
# This should be quick
M = Int(10.0/0.1)
x₀s = range(0.1, 10.0, length=M)
bs = range(0.1, 10.0, length=M)

obj_grid = [mle_obj([x0, b], x, y) for x0 in x₀s, b in bs]

surface(x₀s, bs, obj_grid, xlabel="x₀", ylabel="b", title="Obj Function Landscape", size=(800, 600))
# this does not come out in the way I'd expect and I'm unsure why but need to move on
# the estimation works though!!!



######################################################
# Problem 3
# A lot of this is copy pasted from previous homework

# the main psf calculation that allows for offset
function calc_psf(N, λ, NA, pixel_scale, xc::Float64, yc::Float64)
    # define a center
    cent = (N + 1) / 2

    x = collect(1:N) .- cent
    y = collect(1:N) .- cent

    xv = [pixel_scale * (i - xc / pixel_scale) for i in x] 
    yv = [pixel_scale * (j - yc / pixel_scale) for j in y]

    # create radial distances
    r = [sqrt(x^2 + y^2) for x in xv, y in yv]

    # calculate v
    v = (2π / λ) .* NA .* r

    # calculate the damn psf and account for v==0
    psf = [v == 0 ? 1.0 : 4 * (besselj(1, v) / v)^2 for v in v]

    return psf
end;

# A function that pixelates the psf based on fine scaling
function pixelate_psf(N::Int, 
    λ::Float64, 
    aper::Float64, 
    cam_scale::Float64, 
    fine_scale::Float64,
    xc::Float64,
    yc::Float64)

    # need to calculate the psf for a fine grid
    fine_N = Int(N * cam_scale / fine_scale)
    fine_psf = calc_psf(fine_N, λ, aper, fine_scale, xc, yc)

    # down sample that feller
    coarse_psf = zeros(N, N) # initiate
    block_size = Int(cam_scale / fine_scale) # define the block size

    for i=1:1:N, j=1:1:N # iterate through using block sizes
        fine_i = ((i - 1) * block_size + 1):(i * block_size)
        fine_j = ((j - 1) * block_size + 1):(j * block_size)
        coarse_psf[i, j] = sum(fine_psf[fine_i, fine_j])
    end

    return coarse_psf / sum(coarse_psf) # return the normalized version
end;

# add noise to the psf itself
function add_noise(psf_array, Nₚ)
    # scale the psf so that the total intensity = Nₚ
    scaled_psf = psf_array * Nₚ

    # add noise
    noisy_psf = [rand(Poisson(intensity)) for intensity in scaled_psf]

    return noisy_psf
end;

# add background noise
function bg_noise(psf_array, N, bg)
    noise = [rand(Poisson(bg)) for _ in 1:1:N, _ in 1:1:N]

    final_image = [psf_array[i, j] + noise[i, j] for i in 1:1:N, j in 1:1:N]

    return final_image
end;

# Put all the previous functions together and return a noisy psf able to be offset and add bg noise
function psf(N, λ, NA, camera_scale, fine_scale, Nₚ, xc, yc, bg)
    coarse_psf = pixelate_psf(N, λ, NA, camera_scale, fine_scale, xc, yc)
    noisy_psf = add_noise(coarse_psf, Nₚ)
    final_psf = bg_noise(noisy_psf, N, bg)

    return final_psf
end;

# Centroid calc
function calc_centroid(img)
    # Extract image dimensions
    N, M = size(img)

    # create coord grid for x and y
    x, y = 1:N, 1:M
    xv = repeat(x', M, 1)  # transpose to match dimensions
    yv = repeat(y, 1, N)

    # calculate weighted sum for centroid
    I_total = sum(img)
    xc = sum(xv .* img) / I_total
    yc = sum(yv .* img) / I_total

    return xc, yc
end;

# New parameters
N = 7;
λ = 0.510;
NA = 0.9;
Nₚ = 1000;
camera_scale = 0.1;
fine_scale = 0.01;
bg = 10;
M = 100;

# rando numbers between -0.05 and 0.05
x_poss = rand(M) .* 0.1 .- 0.05
y_poss = rand(M) .* 0.1 .- 0.05

# generate sim images
sim_imgs = []

for i = 1:1:M
    push!(sim_imgs, psf(N, λ, NA, camera_scale, fine_scale, Nₚ, x_poss[i], y_poss[i], bg))
end

heatmap(sim_imgs[37]) # check to make sure we have images!!!!


# measure times for centroid localization
cents = []
@time begin
    for i = 1:1:length(sim_imgs)
        push!(cents, calc_centroid(sim_imgs[i]))
    end
end

# Roughly 0.12ms



######################################################
# Problem 4
# Starting over with fresh functions to be safe

function gauss_psf(x, y, A₀, xc, yc, σ, B)
    A = A₀ * exp(-((x - xc)^2 + (y - yc)^2) / (2 * σ^2)) + B

    return A
end;

function neg_log(params, img, x_grid, y_grid)
    A₀, xc, yc, σ, B = params
    
    # calculate expected intensities for each pixel
    exp_Is = []
    exp_Is = gauss_psf.(x_grid, y_grid, A₀, xc, yc, σ, B)
    # for i = 1:1:length(x_grid), j = 1:1:length(y_grid)
    #     push!(exp_Is, gauss_psf(x_grid[i], y_grid[j], A₀, xc, yc, σ, B))
    # end

    # negative log-likelihood assuming Poisson noise
    # log_likelihood = sum(exp_Is .- img .* log.(exp_Is .+ 1e-10))
    log_likelihood = sum(exp_Is .- img .* log.(max.(exp_Is, 1e-10) .+ 1e-10))


    return log_likelihood
end;

function mle_localization(img)
    # initial guess
    init_params = [maximum(img), size(img, 1) / 2, size(img, 2) / 2, 1.0, minimum(img)]
    #=  maximum(img) - guess for A₀ is maximum of the image
        sizes - reasonable to guess center
        1.0 - suppose that the standard deviation could be 1.0
        minimum(img) - guess for B which is background noise =#

    # create dang grids
    rows, cols = size(img)
    x_grid = repeat(collect(1:cols)', rows, 1)
    y_grid = repeat(collect(1:rows), 1, cols)

    # run optimization using Optim packagee
    result = optimize(params -> neg_log(params, img, x_grid, y_grid), init_params)

    # extract the optimized parameters
    A₀, xc, yc, σ, B = Optim.minimizer(result)

    return (xc, yc, A₀, σ, B)
end;

# we already have simulated images so lets try the mle_localization on
# that array so that the comparison is logical
mle_results = []
@time begin
    for i = 1:1:length(sim_imgs)
        push!(mle_results, mle_localization(sim_imgs[i]))
    end
end



######################################################
# Problem 4c

# New parameters
N = 7;
λ = 0.510;
NA = 0.9;
Nₚ = 1000;
camera_scale = 0.1;
fine_scale = 0.01;
bg = 0;
M = 100;

# New array for emitter positions
pos = [(0.0, 0.0), (0.3, 0.0), (-0.3, 0.0)]

# Store results for each position
results = Dict{Tuple{Float64, Float64}, Vector{Float64}}()

# Loop over each emitter position
for (x0, y0) in pos
    delta_xs = Float64[]
    
    # Simulate images and apply MLE localization
    for i = 1:1:M
        img = psf(N, λ, NA, camera_scale, fine_scale, Nₚ, x0, y0, bg)
        xc, yc, _, _, _ = mle_localization(img)
        
        # Calculate localization error in x
        push!(delta_xs, xc - x0)
    end
    
    # Store the error values for this position
    results[(x0, y0)] = delta_xs
end

for (pos, delta_xs) in results
    histo = histogram(delta_xs, bins=90, 
                xlabel="Δx (pixels)", 
                ylabel="frequency", 
                title="histogram of Δx for position $(pos)", 
                label=false)
    display(current())
end


######################################################
# Problem 4d

# Diagonal positions from (-0.5, -0.5) to (0.5, 0.5) in steps of 0.1
positions = [(p, p) for p in -0.5:0.1:0.5]
mean_deltas = Float64[]

# loop across the diagonal
for (x0, y0) in positions
    delta_xs = Float64[]
    
    # Simulate M images for the current position and calculate localization error
    for i = 1:1:M
        img = psf(N, λ, NA, camera_scale, fine_scale, Nₚ, x0, y0, bg)
        xc, yc, _, _, _ = mle_localization(img)
        
        # Calculate localization error in x
        push!(delta_xs, xc - x0)
    end
    
    # Calculate and store the mean Δx for this position
    push!(mean_deltas, mean(delta_xs))
end

p_values = [pos[1] for pos in positions]

last_plot = plot(p_values, mean_deltas, xlabel="position (p)", ylabel="mean Δx (pixels)",
     title="mean Δx vs. position", marker=:o, legend=false)

savefig(last_plot, "homework/submitted/homework_6/images/last_plot.png")