module ExamplePkg

using PlutoUI, PlutoTeachingTools, Random, Statistics, BenchmarkTools
using Distances, LinearAlgebra, StatsBase, PyCall, LsqFit
using LazyArrays, OhMyThreads, Test

const scipy = pyimport("scipy")

using Markdown
using InteractiveUtils

# ╔═╡ 4ae96412-738d-47b9-ac3b-564f64899c7e
using FITSIO, Tables

export GalaxyCatalog
export fit_overdensity_field_parallel

# ╔═╡ 9d262042-8910-460c-9007-2371f119467e
md"**Catalog structure**"

# ╔═╡ ca2c1744-a07f-11f0-1d83-21b6e4d93d48
begin
	# Define catalog structure
	struct GalaxyCatalog
	    ra::Vector{Float64}
	    dec::Vector{Float64}
	    redshift::Vector{Float64}
	    n_galaxies::Int
	    cosmology::NamedTuple
	end

	# Initialization function for GalaxyCatalog
	function GalaxyCatalog(
	    ra::AbstractVector{T1},
	    dec::AbstractVector{T2},
	    redshift::AbstractVector{T3};
	    cosmology_input::Union{NamedTuple, Nothing}=nothing
	) where {T1 <: Number, T2 <: Number, T3 <: Number}
	    # Assert that all input vectors have the same length
	    @assert length(ra) == length(dec) == length(redshift) "Input arrays ra, dec, and redshift must have the same length."
	
	    n_galaxies = length(ra)
	
	    # Handle the default cosmology
	    final_cosmology = if isnothing(cosmology_input)
	        (H0=70.0, Om0=0.3, Ode0=0.7)
	    else
	        cosmology_input
	    end
	
	    # Create the instance, ensuring the data is Float64 to match the struct definition
	    return GalaxyCatalog(Float64.(ra), Float64.(dec), Float64.(redshift), n_galaxies, final_cosmology)
	end
end;

# ╔═╡ c4b0c635-e049-4a4f-b7ff-9c63df9ad263
md"**Convert to comoving coordinates.**"

# ╔═╡ 1d3c6186-aad6-447b-9014-7940ac67235d
md"**Estimate volume**"

# ╔═╡ 589607f1-9e16-4bc6-871b-7afef9374d97
md"**Calculate ξ(r,z) - the 2D correlation function.**"

# ╔═╡ 60916f89-23d6-4a8b-a649-d306d4b00b5f
md"**Calculate correlation function for a single redshift slice**"

# ╔═╡ 5fc9beb0-3a17-486d-a7cc-360265653fa2
md"**Fit parametric models to ξ(r,z).**"

# ╔═╡ 2cfe2c1c-979d-4b05-b8a9-da23dbc6dabe
function
	simple_power_law_model_rp(r::AbstractVector{T1}, p::AbstractVector{T2}
) where {T1 <: Number, T2 <: Number}
	return p[1] .* (r ./ p[2]) .^ (-p[3])
end;

# ╔═╡ 301e3c97-509b-4c4c-a6a9-e4643bdda692
begin
	function power_law_single(r::T1, z::T2, p::AbstractVector{T3}
	) where {T1 <: Number, T2 <: Number, T3 <: Number}
		A0, r0_0, gamma, alpha_A, alpha_r = p
		A_z  = A0  * (1 + z) ^ alpha_A
    	r0_z = r0_0 * (1 + z) ^ alpha_r
    	return A_z * (r / r0_z) ^ (-gamma)
	end

	function cutoff_single(r::T1, z::T2, p::AbstractVector{T3}
	) where {T1 <: Number, T2 <: Number, T3 <: Number}
	    A0, r0_0, gamma, rc, alpha = p
	    A_z  = A0  * (1 + z) ^ alpha
	    r0_z = r0_0 + 0 * z
	    return A_z * (r / r0_z) ^ (-gamma) * exp.(-r / rc)	
	end
end;

# ╔═╡ 50a19555-b784-4ed0-ba0d-861c76a12d89
function power_law_func_rp(X::AbstractMatrix{T1}, p::AbstractVector{T2}
) where {T1 <: Number, T2 <: Number}
    r = @view(X[1, :])
    z = @view(X[2, :])
    return power_law_single.(r, z, p)
end;

# ╔═╡ a5a7a147-3ada-4e85-9da3-d3e1d99a8b4f
function cutoff_func_rp(X::AbstractMatrix{T1}, p::AbstractVector{T2}
) where {T1 <: Number, T2 <: Number}
    r = @view(X[1, :])
    z = @view(X[2, :])
    A0, r0_0, gamma, rc, alpha = p
    return cutoff_single.(r, z, p)
end;

# ╔═╡ 00df1319-b35c-492a-be95-3804ec2da556
function fit_correlation_models(
    r_centers::AbstractVector{T1}, 
    z_centers::AbstractVector{T2}, 
    xi_2d::AbstractMatrix{T3}
) where {T1 <: Number, T2 <: Number, T3 <: Number}
	# r_flat: repeats each element of r_centers (M = length(z_centers)) times.
	r_flat = repeat(r_centers, inner = length(z_centers))

	# z_flat: repeats the entire z_centers vector (N = length(r_centers)) times.
	z_flat = repeat(z_centers, outer = length(r_centers))

	xi_flat = vec(xi_2d)
	
	valid_mask= isfinite.(xi_flat) .& (xi_flat .> 0) .& (r_flat .> 0)
	r_fit = @view r_flat[valid_mask]
	z_fit = @view z_flat[valid_mask]
	xi_fit = @view xi_flat[valid_mask]
	
	if length(xi_fit) < 10
	    return nothing, nothing, nothing
	end
	simple_fits = []
	p0_simple_power_law = [1.0, 5.0, 1.8]
	
	for (i, z_center) in enumerate(z_centers)
		# Check if *any* data in this column is positive.
	    if any(@view(xi_2d[:, i]) .> 0)
	        valid_r = @view(xi_2d[:, i]) .> 0
	        r_slice = @view r_centers[valid_r]
	        xi_slice = @view xi_2d[valid_r, i]

			# Fit if we have at least 3 valid points
	        if length(xi_slice) >= 3
	                popt = curve_fit(simple_power_law_model_rp, r_slice, xi_slice, p0_simple_power_law)
	                push!(simple_fits, (z_center, popt))
	        end
	    end
	end
	
	X = vcat(permutedims(r_fit), permutedims(z_fit))
	p0_power = [1.0, 5.0, 1.8, 0.0, 0.0]
	fit_power = curve_fit(power_law_func_rp, X, xi_fit, p0_power)
	p0_cutoff = [1.0, 5.0, 1.8, 20.0, 0.0]
	fit_cutoff = curve_fit(cutoff_func_rp, X, xi_fit, p0_cutoff)
	return simple_fits, coef(fit_power), coef(fit_cutoff)
end;

# ╔═╡ 972d2dfd-310d-4acd-a0ea-46c043876415
md"**Complete pipeline to fit δP = N̄[1 + ξ(r,z)]δV**"

# ╔═╡ 733e2c72-772c-495f-88b4-4bff9266d6cc
md"**Read real galaxy data**"

# ╔═╡ 0c196352-0793-4ddb-8ea0-81300db2b7d7
"""
	read_galaxy_fits(filename::String; cosmology_input=nothing)

	Read a galaxy catalog from a FITS file and construct a `GalaxyCatalog`.

	The FITS file must contain columns:
	    "GField", "RA", "DEC", "redshift"
"""
function read_galaxy_fits(filename::String; cosmology_input=nothing)
	    f = FITS(filename)
	    # The table is stored in the 2nd HDU
	    hdu = f[2]

	    # Get column names
	    colnames = FITSIO.colnames(hdu)
	    lowercols = lowercase.(colnames)

	    # Helper to find column ignoring case
	    function getcol(target)
	        idx = findfirst(==(lowercase(target)), lowercols)
	        @assert idx !== nothing "Column '$target' not found in FITS file."
	        return read(hdu, colnames[idx])
	    end

		# Read the ra, dec, redshift data
	    ra = Float64.(getcol("ra"))
	    dec = Float64.(getcol("dec"))
	    redshift = Float64.(getcol("redshift"))

	    close(f)

	    return GalaxyCatalog(ra, dec, redshift; cosmology_input=cosmology_input)
	end

# ╔═╡ d5174299-c36c-42b9-a02e-c14806734ab5
md"**Code for calculate the distance and then bin distance**"

# ╔═╡ 1f5f8f8a-5ebc-44ed-a5aa-e6e1ac092106
begin
"""
	bin_function(d,edges)

	return the value in edges that just smaller than d

	for d=1, deges=[0.2,1.2,2.2], return 1 as 0.2<1<1.2

"""
@inline function bin_function(d::Float64, edges::AbstractVector{T}
) where {T <: Number}
    i = searchsortedlast(edges, d)
    return i
end

"""
	bin_data(hist,P,r_edges)

	calculate the 3D distance and then bin the distance for each elements in P, return the hist plot based on bin value.

"""
function bin_data!(
    hist::AbstractVector{T1},
    P::AbstractMatrix{T2},
    r_edges::AbstractVector{T3}
) where {T1 <: Number, T2 <: Number, T3 <: Number}
    N = size(P, 1)
	rmax2 = (last(r_edges))^2  #set the boundary
    @inbounds for i in 1:N-1
        xi, yi, zi = P[i,1], P[i,2], P[i,3]
        for j in (i+1):N
            dx = P[j,1]-xi; dy = P[j,2]-yi; dz = P[j,3]-zi
            r2 = dx*dx + dy*dy + dz*dz
			if r2<= rmax2 # remove r2 that larger than max boundary
            	b = bin_function(sqrt(r2), r_edges)
            	if b!=0  # remove bin that is related r2< min boundary
					hist[b] += 1.0
				end
			end
        end
    end
    return hist
end

"""
	bin_data_cross(hist,P,r_edges)

	calculate the 3D distance and then bin the distance for each elements in P and Q, return the hist plot based on bin value.

"""
function bin_data_cross!(
    hist::AbstractVector{T1},
    P::AbstractMatrix{T2},
    Q::AbstractMatrix{T3},
    r_edges::AbstractVector{T4}
) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: Number}
    N, M = size(P,1), size(Q,1)
	rmax2 = (last(r_edges))^2 #set the boundary
    @inbounds for i in 1:N
        xi, yi, zi = P[i,1], P[i,2], P[i,3]
        for j in 1:M
            dx = Q[j,1]-xi; dy = Q[j,2]-yi; dz = Q[j,3]-zi
            r2 = dx*dx + dy*dy + dz*dz
			if r2<= rmax2 # remove r2 that larger than max boundary
				b = bin_function(sqrt(r2), r_edges)
				if b!=0 # remove bin that is related r2< min boundary
					hist[b] += 1.0
				end
			end
        end
    end
    return hist
end
end

# ╔═╡ bcfbdec0-e5e0-4a6e-8c4d-80ebae0dceeb
function calculate_correlation_slice!(
    xi_slice_output::AbstractVector{T1},
    positions::AbstractMatrix{T2},
    r_bins::AbstractVector{T3}
) where {T1 <: Number, T2 <: Number, T3 <: Number}

    n_gal = size(positions, 1)
    nb = length(r_bins) - 1
    @assert length(xi_slice_output) == nb

    DD = zeros(Float64, nb)
    bin_data!(DD, positions, r_bins)

    n_random = min(max(2*n_gal, 5_000), 10_000)

    min_coords = minimum(positions, dims=1)
    max_coords = maximum(positions, dims=1)
    random_pos = rand(n_random, 3) .* (max_coords .- min_coords) .+ min_coords

    RR = zeros(Float64, nb)
    bin_data!(RR, random_pos, r_bins)

    DR = zeros(Float64, nb)
    bin_data_cross!(DR, positions, random_pos, r_bins)

    # --- Normalize (same as your code) ---
    DD ./= (n_gal * (n_gal - 1) / 2)
    RR ./= (n_random * (n_random - 1) / 2)
    DR ./= (n_gal * n_random)

    # Landy–Szalay
    @inbounds for b in 1:nb
        xi_slice_output[b] = RR[b] > 0 ? (DD[b] - 2*DR[b] + RR[b]) / RR[b] : 0.0
    end
    return nothing
end

# ╔═╡ 1f937d4b-6be0-49c5-a515-d5db4fa7b8d5
md"# Parallel version of code"

# ╔═╡ e40902d4-f9b0-4072-8bfb-4577db3d5e7b

begin
"""
	parallel version of bin_function(d,edges)

	assign one histogram for each thread and then combine togher.

	return the value in edges that just smaller than d

	for d=1, deges=[0.2,1.2,2.2], return 1 as 0.2<1<1.2

"""
function bin_data_parallel!(
    hist::AbstractVector{T1},
    P::AbstractMatrix{T2},
    r_edges::AbstractVector{T3}
) where {T1 <: Number, T2 <: Number, T3 <: Number}
    N  = size(P, 1)
    nb = length(r_edges) - 1
    rmax2 = (last(r_edges))^2

    
    hloc = [zeros(eltype(hist), nb) for u in 1:Threads.nthreads()] # each thread will got one hist

	Threads.@threads for i in 1:N-1
        hi = hloc[Threads.threadid()]
        xi, yi, zi = P[i,1], P[i,2], P[i,3]
        @inbounds for j in (i+1):N
            dx = P[j,1]-xi; dy = P[j,2]-yi; dz = P[j,3]-zi
            r2 = dx*dx + dy*dy + dz*dz
            if r2 <= rmax2
				b = bin_function(sqrt(r2), r_edges)
                if b != 0
                    hi[b] += 1.0
                end
            end
        end
    end
    
    fill!(hist, zero(eltype(hist))) # combine them togher
    @inbounds for h in hloc
        hist .+= h
    end
    return hist
end

function bin_data_cross_parallel!(
    hist::AbstractVector{T1},
    P::AbstractMatrix{T2},
    Q::AbstractMatrix{T3},
    r_edges::AbstractVector{T4}
) where {T1 <: Number, T2 <: Number, T3 <: Number, T4 <: Number}
    N, M = size(P,1), size(Q,1)
    nb = length(r_edges) - 1
    rmax2 = (last(r_edges))^2
	
    hloc = [zeros(eltype(hist), nb) for u in 1:Threads.nthreads()]

	Threads.@threads for i in 1:N-1
        hi = hloc[Threads.threadid()]
        xi, yi, zi = P[i,1], P[i,2], P[i,3]
        @inbounds for j in 1:M
            dx = Q[j,1]-xi; dy = Q[j,2]-yi; dz = Q[j,3]-zi
            r2 = dx*dx + dy*dy + dz*dz
            if r2 <= rmax2
                b = bin_function(sqrt(r2), r_edges)
                if b != 0
                    hi[b] += 1.0
                end
            end
        end
    end

    fill!(hist, zero(eltype(hist)))
    @inbounds for h in hloc
        hist .+= h
    end
    return hist
end
end

# ╔═╡ 94692b18-e3e7-4590-bade-4d173326a32c
function calculate_correlation_slice_parallel!(
    xi_slice_output::AbstractVector{T1},
    positions::AbstractMatrix{T2},
    r_bins::AbstractVector{T3}
) where {T1 <: Number, T2 <: Number, T3 <: Number}

    n_gal = size(positions, 1)
    nb = length(r_bins) - 1
    @assert length(xi_slice_output) == nb

	DD = zeros(Float64, nb)
	bin_data_parallel!(DD, positions, r_bins)

    n_random = min(max(2*n_gal, 5_000), 10_000)

    min_coords = minimum(positions, dims=1)
    max_coords = maximum(positions, dims=1)
    random_pos = rand(n_random, 3) .* (max_coords .- min_coords) .+ min_coords

	RR = zeros(Float64, nb)
	bin_data_parallel!(RR, random_pos, r_bins)

	DR = zeros(Float64, nb)
	bin_data_cross_parallel!(DR, positions, random_pos, r_bins)

    # --- Normalize (same as your code) ---
    DD ./= (n_gal * (n_gal - 1) / 2)
    RR ./= (n_random * (n_random - 1) / 2)
    DR ./= (n_gal * n_random)

    # Landy–Szalay
    @inbounds for b in 1:nb
        xi_slice_output[b] = RR[b] > 0 ? (DD[b] - 2*DR[b] + RR[b]) / RR[b] : 0.0
    end
    return nothing
end

# ╔═╡ 223ea73f-6b25-4778-9ef8-5d5e49759ea6
md"# Constants and Packages"

# ╔═╡ c38597eb-1f51-4cd3-8edb-6b6e9e914a45
# # Packages
# begin
# 	using Pkg
	
# 	# 1. Activate local environment and set up Python for PyCall
# 	Pkg.activate(mktempdir())
# 	ENV["PYTHON"] = "" # Force PyCall to use its own Conda-managed Python
	
# 	# 2. Add all Julia and Python packages
# 	# Using Pkg.add() here ensures that if you add a new package 
# 	# (like LazyArrays), it will be automatically added to your project.
# 	Pkg.add([
# 	    "PlutoUI", "PlutoTest", "PlutoTeachingTools", 
# 	    "BenchmarkTools", "Distances", "StatsBase",
# 	    "PyCall", "Conda", "LsqFit", "FITSIO", "Tables",
#         "LazyArrays", "OhMyThreads" 
# 	])
	
# 	# 3. Load the packages and import the Python library
# 	# Added all necessary packages to the using statement
# 	using PlutoUI, PlutoTest, PlutoTeachingTools, Random, Statistics, BenchmarkTools
# 	using Distances, LinearAlgebra, StatsBase, PyCall, LsqFit
# 	using LazyArrays 
# 	using OhMyThreads
	
# 	const scipy = pyimport("scipy")
	
# 	println("Setup complete. SciPy and other packages are ready to use.")
# end;

# ╔═╡ 1ba18d8f-241e-418b-a2d1-eb388fe26925
# Constants
begin
	c = 299792.458  # km/s
end

# ╔═╡ 3dec3c42-d1a1-4121-8b83-638d9678619b
begin
	# Works with a single redshift and H0
	function comoving_distance(z::T1, H0::T2) where {T1 <: Number, T2 <: Number}
	    if z < 0.3
	        return c * z / H0
	    else
	        return (c * z / H0) * (1 + z / 2)
	    end
	end
end;

# ╔═╡ 873b10f4-d53f-472a-9ac9-c974f657d73c
function convert_to_comoving(catalog::GalaxyCatalog)
    dc = @~ comoving_distance.(catalog.redshift, catalog.cosmology.H0)
    
    x = @~ dc .* cosd.(catalog.dec) .* cosd.(catalog.ra)
    y = @~ dc .* cosd.(catalog.dec) .* sind.(catalog.ra)
    z = @~ dc .* sind.(catalog.dec)
    
    return @~ hcat(x, y, z)
end;


# ╔═╡ e2f6d3e5-ce3c-4952-967f-62d8bca15305
"""
Calculate ξ(r,z) - the 2D correlation function.

	Parameters:
	-----------
	r_bins : array-like
			Radial separation bins (Mpc/h)
	z_bins : array-like
			Redshift bins
			
	Returns:
	--------
	r_centers, z_centers : arrays
			Bin centers
	xi_2d : 2D array
			Correlation function ξ(r,z)
"""
function calculate_2d_correlation_function(catalog::GalaxyCatalog, r_bins::AbstractVector{T1}, z_bins::AbstractVector{T2}
) where {T1 <: Number, T2 <: Number}
    println("Calculating 2D correlation function xi(r,z)...")
    
    # Get comoving positions for the entire catalog
    comoving_positions = convert_to_comoving(catalog)

    # Create bin centers
    r_centers = @. (@view(r_bins[1:end-1]) + @view(r_bins[2:end])) / 2
    z_centers = @. (@view(z_bins[1:end-1]) + @view(z_bins[2:end])) / 2
    
    # Initialize correlation function array
    xi_2d = zeros(length(r_centers), length(z_centers))
    
    # Calculate for each redshift slice
    for (i, z_center) in enumerate(z_centers)
        println("Processing redshift bin $i/$(length(z_centers)): z=$(round(z_center, digits=3))")
        
        # Select galaxies in this redshift bin (using broadcasting)
        z_indices = findall(z -> z_bins[i] <= z < z_bins[i+1], catalog.redshift)
        
        num_in_bin = length(z_indices)
        
        num_in_bin = sum(z_indices)
        if num_in_bin < 10  # Skip if too few galaxies
            println("  Warning: Only $num_in_bin galaxies in this z-bin")
            continue
        end
        
        # Get positions for this redshift slice
		# comoving_positions is a lazy array
        positions_z = comoving_positions[z_indices, :]
        
        # Calculate correlation function for this slice
        if size(positions_z, 1) > 5
			xi_slice_output_buffer = @view xi_2d[:, i]
            calculate_correlation_slice!(xi_slice_output_buffer, positions_z, r_bins)
        end
    end
    
    return r_centers, z_centers, xi_2d
end;

# ╔═╡ c272fef3-6cff-4869-a848-f8c74a342401
function fit_overdensity_field(catalog::GalaxyCatalog; r_bins::AbstractVector{T1} = nothing, z_bins::AbstractVector{T2} = nothing) where {T1 <: Number, T2 <: Number}
	n_galaxies = catalog.n_galaxies
	redshift = catalog.redshift
	
	if r_bins === nothing
	    r_bins  = logrange(1, 100, length=12)
	end
	if z_bins === nothing
	    n_z_bins = min(8, max(3, floor(Int,n_galaxies/100)))  # Adaptive binning
	    z_bins = LinRange(minimum(redshift), maximum(redshift), n_z_bins + 1)
	end
	r_centers, z_centers, xi_2d = calculate_2d_correlation_function(catalog, r_bins, z_bins)
	simple_fits, popt_power, popt_cutoff = fit_correlation_models(r_centers, z_centers, xi_2d)
	
	return (r_centers=r_centers, z_centers=z_centers, xi_2d=xi_2d, simple_fits=simple_fits, power_law_params=popt_power, cutoff_params=popt_cutoff)
end;

# ╔═╡ b3fa0365-5aed-4415-8eaa-ab3c89801996
"""
Calculate ξ(r,z) - parallelzed the 2D correlation function. just change calculate_2d_correlation_function to calculate_2d_correlation_function_parallel

	Parameters:
	-----------
	r_bins : array-like
			Radial separation bins (Mpc/h)
	z_bins : array-like
			Redshift bins
			
	Returns:
	--------
	r_centers, z_centers : arrays
			Bin centers
	xi_2d : 2D array
			Correlation function ξ(r,z)
"""
function calculate_2d_correlation_function_parallel(catalog::GalaxyCatalog, r_bins::AbstractVector{T1}, z_bins::AbstractVector{T2}
) where {T1 <: Number, T2 <: Number}
    println("Calculating 2D correlation function xi(r,z)...")
    
    # Get comoving positions for the entire catalog
    comoving_positions = convert_to_comoving(catalog)

    # Create bin centers
    r_centers = @. (@view(r_bins[1:end-1]) + @view(r_bins[2:end])) / 2
    z_centers = @. (@view(z_bins[1:end-1]) + @view(z_bins[2:end])) / 2
    
    # Initialize correlation function array
    xi_2d = zeros(length(r_centers), length(z_centers))
    
    # Calculate for each redshift slice
    for (i, z_center) in enumerate(z_centers)
        println("Processing redshift bin $i/$(length(z_centers)): z=$(round(z_center, digits=3))")
        
        # Select galaxies in this redshift bin (using broadcasting)
        z_indices = findall(z -> z_bins[i] <= z < z_bins[i+1], catalog.redshift)
        
        num_in_bin = length(z_indices)
        
        num_in_bin = sum(z_indices)
        if num_in_bin < 10  # Skip if too few galaxies
            println("  Warning: Only $num_in_bin galaxies in this z-bin")
            continue
        end
        
        # Get positions for this redshift slice
		# comoving_positions is a lazy array
        positions_z = comoving_positions[z_indices, :]
        
        # Calculate correlation function for this slice
        if size(positions_z, 1) > 5
			xi_slice_output_buffer = @view xi_2d[:, i]
            calculate_correlation_slice_parallel!(xi_slice_output_buffer, positions_z, r_bins)
        end
    end
    
    return r_centers, z_centers, xi_2d
end;

# ╔═╡ 9e710409-8780-458f-bdf2-3c9bffcbb33d
"""
fit_overdensity_field_parallel: parallelzed version of fit_overdensity_field, only chaned calculate_2d_correlation_function to calculate_2d_correlation_function_parallel.
"""
function fit_overdensity_field_parallel(catalog::GalaxyCatalog; r_bins::AbstractVector{T1} = nothing, z_bins::AbstractVector{T2} = nothing) where {T1 <: Number, T2 <: Number}
	n_galaxies = catalog.n_galaxies
	redshift = catalog.redshift
	
	if r_bins === nothing
	    r_bins  = logrange(1, 100, length=12)
	end
	if z_bins === nothing
	    n_z_bins = min(8, max(3, floor(Int,n_galaxies/100)))  # Adaptive binning
	    z_bins = LinRange(minimum(redshift), maximum(redshift), n_z_bins + 1)
	end
	r_centers, z_centers, xi_2d = calculate_2d_correlation_function_parallel(catalog, r_bins, z_bins)
	simple_fits, popt_power, popt_cutoff = fit_correlation_models(r_centers, z_centers, xi_2d)
	
	return (r_centers=r_centers, z_centers=z_centers, xi_2d=xi_2d, simple_fits=simple_fits, power_law_params=popt_power, cutoff_params=popt_cutoff)
end;

# ╔═╡ 37c3d937-ab8d-4de9-9e78-bc4a63ee4afd
# Estimate the survey volume
function estimate_survey_volume(catalog::GalaxyCatalog)
	# Angular area in steradians
    ra_range = maximum(catalog.ra) - minimum(catalog.ra)
    dec_range = maximum(catalog.dec) - minimum(catalog.dec)
    
    area_deg2 = ra_range * dec_range * cosd(mean(catalog.dec))
    area_sr = area_deg2 * (pi / 180)^2
    
    # Radial distances
    d_min = comoving_distance(minimum(catalog.redshift), catalog.cosmology.H0)
    d_max = comoving_distance(maximum(catalog.redshift), catalog.cosmology.H0)
    
    # Calculate volume as a sector of a spherical shell
    volume = (area_sr / 3.0) * (d_max^3 - d_min^3)
    
    return volume
end;



end # module ExamplePkg
