using Plots;
using SpecialFunctions, Random, Distributions, Statistics;
using Optim;

function meshgrid(x, y)
    X = repeat(x', length(y), 1)
    Y = repeat(y, 1, length(x))
    return X, Y
end;

function sim_img(N, λ, NA, camera_scale, fine_scale, N_photon; center = (0, 0), B = 0)
    block_size = Int(camera_scale / fine_scale)
    fine_N = N * block_size
    
    x = range(-fine_N / 2, fine_N / 2, length=fine_N) .* fine_scale
    y = range(-fine_N / 2, fine_N / 2, length=fine_N) .* fine_scale

    # x = (-fine_N÷2:fine_N÷2 - 1) * fine_scale
    # y = (-fine_N÷2:fine_N÷2 - 1) * fine_scale
    X, Y = meshgrid(x, y)
    
    xc, yc = center
    r = sqrt.((X .- xc).^2 .+ (Y .- yc).^2)
    
    v = (2π / λ) * NA * r
    
    psf_fine = zeros(size(v))
    psf_fine[v .== 0] .= 1
    psf_fine[v .!= 0] .= 4 * (besselj.(1, v[v .!= 0]) ./ v[v .!= 0]).^2
    psf_fine ./= sum(psf_fine)
    
    psf_camera = zeros(N, N)
    
    for i in 1:N
        for j in 1:N
            block = psf_fine[(i-1)*block_size+1:i*block_size, (j-1)*block_size+1:j*block_size]
            psf_camera[i, j] = sum(block)
        end
    end
    
    psf_camera ./= sum(psf_camera)
    psf_camera .*= N_photon
    
    noisy_psf = rand.(Poisson.(psf_camera))
    background = rand.(Poisson(B), N, N)
    
    final_image = noisy_psf .+ background
    
    return final_image
end;

function calc_MLE(intensities, initial_guess; camera_scale=0.01, origin_center=true)
    N = size(intensities, 1)
    x = collect(1:N)
    y = collect(1:N)
    
    if origin_center
        x .-= (N - 1) / 2
        y .-= (N - 1) / 2
    end
    
    X, Y = meshgrid(x, y)

    function mle_obj(params)
        xc, yc, A0, sigma, B = params
        model = A0 * exp.(-((X .- xc).^2 .+ (Y .- yc).^2) / (2 * sigma^2)) .+ B
        sum(model .- intensities .* log.(max.(model, 1)))
    end

    results = optimize(mle_obj, Float64.(initial_guess))
    xc_est, yc_est, A0_est, sigma_est, B_est = Optim.minimizer(results)
    
    if camera_scale != 0
        xc_est *= camera_scale
        yc_est *= camera_scale
    end

    return xc_est, yc_est, A0_est, sigma_est, B_est
end;

function rms_calc(est_list, true_list)
    rms_list = [];

    for i = 1:1:length(est_list)
        diff = (est_list[i] .- true_list[i]).^2
        summ = sum(diff)

        rms = sqrt(summ)

        push!(rms_list, rms)
    end

    return rms_list
end;

N = 7;
λ = 0.510;
NA = 0.9;
N_p = 1000;
camera_scale = 0.1;
fine_scale = 0.001;
bg = 10;
m = 100;
initial_guess = [0, 0, 100, 1, 0];

##########################################################################
# 1a
##########################################################################

true_x_list, true_y_list = [], [];
est_x_list, est_y_list = [],[];
x_error_list, y_error_list = [], [];

for i = 1:1:m
    xc = rand(Uniform(-0.05, 0.05))
    yc = rand(Uniform(-0.05, 0.05))
    cent = (xc, yc)

    img = sim_img(N, λ, NA, camera_scale, fine_scale, N_p, center=cent, B=bg)
    
    MLE = calc_MLE(img, initial_guess; camera_scale=camera_scale, origin_center=true)
    
    x_MLE, y_MLE = MLE[1], MLE[2]
    x_error = (x_MLE - cent[1])
    y_error = (y_MLE - cent[2])

    push!(true_x_list, xc)
    push!(est_x_list, x_MLE)
    push!(x_error_list, x_error)

    push!(true_y_list, yc)
    push!(est_y_list, y_MLE)
    push!(y_error_list, y_error)
end

scatter(true_x_list, x_error_list, 
                    xlabel="true x", 
                    ylabel="∆x", 
                    title="error vs true position", 
                    legend=false, 
                    size=(800, 600))

# savefig(p1a_plt, "homework/submitted/homework_7/1a.png")


##########################################################################
# 1b
##########################################################################
σ = λ / (2 * NA)

photon_list = 10 .^ range(log10(40), log10(40000), length=10);
RMS_list, val_list = [], [];
true_list, est_list = [], [];



for i = 1:1:length(photon_list)
    temp_rmse = []

    # need m images, largely copied from above
    for j = 1:1:m
        xc = rand(Uniform(-0.05, 0.05))
        yc = rand(Uniform(-0.05, 0.05))
        cent = (xc, yc)

        img = sim_img(N, λ, NA, camera_scale, fine_scale, photon_list[i], center=cent, B=bg)
        
        MLE = calc_MLE(img, initial_guess; camera_scale=camera_scale, origin_center=true)
        
        x_MLE, y_MLE = MLE[1], MLE[2]
        Δx = x_MLE - cent[1]
        Δy = y_MLE - cent[2]

        rmse = sqrt(Δx^2 + Δy^2)

        push!(temp_rmse, rmse)
        push!(true_list, cent)
        push!(est_list, (x_MLE, y_MLE))
    end

    mean_rmse = mean(temp_rmse)
    push!(RMS_list, mean_rmse)
    

    val = σ / sqrt(photon_list[i])
    push!(val_list, val)

end

prob1b_plt = plot(photon_list, RMS_list, 
              label = "RMS", 
              xscale=:log10, 
              yscale=:log10, 
              xlabel="N_photons", 
              ylabel="Error", 
              lw=2, 
              size=(800, 600))
plot!(photon_list, val_list, label = "σ / √N_photon", lw=2)

function center_rmse(est_list, true_list)
    est_list = vcat([x for tup in est_list for x in tup]...)
    true_list = vcat([x for tup in true_list for x in tup]...)

    squared_diffs = (est_list .- true_list).^2

    rms_error = sqrt.(mean.(sum.(squared_diffs)))

    return rms_error
    
end




x = rand(Uniform(-0.05, 0.05))
y = rand(Uniform(-0.05, 0.05))
cent = (x, y)
test_img = sim_img(N, λ, NA, camera_scale, fine_scale, N_p, center=cent, B=bg)

heatmap(test_img)

MLE = calc_MLE(test_img, initial_guess; camera_scale=camera_scale, origin_center=true)
x_MLE, y_MLE = MLE[1], MLE[2]
Δx = x_MLE - cent[1]
Δy = y_MLE - cent[2]

rms_error = sqrt(Δx^2 + Δy^2)