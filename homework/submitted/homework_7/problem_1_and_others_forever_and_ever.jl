### LET'S REWRITE THINGS AGAIN SO MUCH FUN LOVE IT
# I LOVE EVERYTHING EVERYTHING IS GREAT

using LinearAlgebra, Random, SpecialFunctions, Distributions

function sim_img(N_camera, wavelength, NA, camera_scale, fine_scale, N_photon; center = (0, 0), B = 0)
    block_size = Int(camera_scale / fine_scale)
    fine_grid_N = N_camera * block_size
    
    x = (-fine_grid_N÷2:fine_grid_N÷2 - 1) * fine_scale
    y = (-fine_grid_N÷2:fine_grid_N÷2 - 1) * fine_scale
    X, Y = meshgrid(x, y)
    
    xc, yc = center
    r = sqrt.((X .- xc).^2 .+ (Y .- yc).^2)
    
    v = (2 * π / wavelength) * NA * r
    
    psf_fine = zeros(size(v))
    psf_fine[v .== 0] .= 1
    psf_fine[v .!= 0] .= 4 * (besselj.(1, v[v .!= 0]) ./ v[v .!= 0]).^2

    
    psf_fine ./= sum(psf_fine)
    
    psf_camera = zeros(N_camera, N_camera)
    
    for i in 1:N_camera
        for j in 1:N_camera
            block = psf_fine[(i-1)*block_size+1:i*block_size, (j-1)*block_size+1:j*block_size]
            psf_camera[i, j] = sum(block)
        end
    end
    
    psf_camera ./= sum(psf_camera)
    psf_camera .*= N_photon
    
    noisy_psf = rand.(Poisson.(psf_camera))
    background = rand.(Poisson(B), N_camera, N_camera)
    
    final_image = noisy_psf .+ background
    
    return final_image
end;

using Optim, Statistics

function calculate_MLE(intensities, initial_guess; camera_scale=0.01, origin_center=true)
    N = size(intensities, 1)
    x = collect(1:N)
    y = collect(1:N)
    
    if origin_center
        x .-= (N - 1) / 2
        y .-= (N - 1) / 2
    end
    
    X, Y = meshgrid(x, y)

    function objective(params)
        xc, yc, A0, sigma, B = params
        model = A0 * exp.(-((X .- xc).^2 .+ (Y .- yc).^2) / (2 * sigma^2)) .+ B
        sum(model .- intensities .* log.(max.(model, 1)))
    end

    results = optimize(objective, initial_guess)
    xc_est, yc_est, A0_est, sigma_est, B_est = Optim.minimizer(results)
    
    if camera_scale != 0
        xc_est *= camera_scale
        yc_est *= camera_scale
    end

    return xc_est, yc_est, A0_est, sigma_est, B_est
end;


function calculate_rms_error(calculated_positions, true_positions)
    # Ensure arrays are 2D, even if only one position is provided
    if length(calculated_positions) == 2
        calculated_positions = hcat(calculated_positions...)
    end
    if length(true_positions) == 2
        true_positions = hcat(true_positions...)
    end
    
    squared_diffs = (calculated_positions .- true_positions).^2
    rms_error = sqrt(mean(sum(squared_diffs, dims=2)))
    
    return rms_error
end;

function meshgrid(x, y)
    X = repeat(x', length(y), 1)
    Y = repeat(y, 1, length(x))
    return X, Y
end;

using Random, Plots

# Define parameters
N_camera = 7
wavelength = 0.510
NA = 0.9
N_photons = 1000
camera_scale = 0.1
fine_scale = 0.001
bg = 10
m = 1000
initial_guess = [0, 0, 100, 1, 0]

true_x_list = Float64[]
x_error_list = Float64[]

# Simulation loop
for _ in 1:m
    center_x = rand(Uniform(-0.05, 0.05))
    center_y = rand(Uniform(-0.05, 0.05))
    current_center = (center_x, center_y)

    image = sim_img(N_camera, wavelength, NA, camera_scale, fine_scale, N_photons, center=current_center, B=bg)
    
    MLE = calculate_MLE(image, initial_guess, camera_scale=camera_scale, origin_center=true)
    
    x_MLE = MLE[1]
    x_error = x_MLE - current_center[1]

    push!(true_x_list, center_x)
    push!(x_error_list, x_error)
end

# Plotting
scatter(true_x_list, x_error_list, xlabel="True X-position (μm)", ylabel="∆X (μm)", legend=false)