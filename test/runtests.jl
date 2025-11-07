using Test
using ExamplePkg
using Random

@testset "Complete Overdensity Field Fitting" begin
    Random.seed!(321)
    n_gals = 200
    catalog = GalaxyCatalog(
        rand(n_gals) .* 360.0,
        rand(n_gals) .* 180.0 .- 90.0,
        rand(n_gals) .* 2.0
    )
    
    r_bins = 10 .^ range(0, 2, length=10)
    z_bins = range(minimum(catalog.redshift), maximum(catalog.redshift), length=4)
    
    result = fit_overdensity_field_parallel(catalog; r_bins=r_bins, z_bins=z_bins)
    
    @test haskey(result, :r_centers)
    @test haskey(result, :z_centers)
    @test haskey(result, :xi_2d)
    @test all(isfinite.(result.r_centers))
end