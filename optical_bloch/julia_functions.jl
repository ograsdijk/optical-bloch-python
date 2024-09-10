using Distributed

@everywhere begin
    """
        phase_modulation(t::Float64, β::Float64, ω::Float64)::ComplexF64

    Compute phase modulation at frequency ω with a modudulation strength β at time t
    """
    function phase_modulation(t::Float64, β::Float64, ω::Float64)::ComplexF64
        return exp(1im.*β.*sin(ω.*t))
    end

    """
        resonant_switching(t::Float64, ω::Float64, phase::Float64)

    generate the polarization switching coming from a resonant EOM
    """
    function resonant_switching(t::Float64, ω::Float64, phase::Float64)::Float64
        -cos(pi*(1 .+ cos(ω .* t .+ phase))/2)/2 + 1/2
    end

    """
        gaussian(x::Float64, a::Float64, μ::Float64, σ::Float64)

    generate a gaussian for amplitude a, mean μ and sigma σ
    """
    function gaussian(x::Float64, a::Float64, μ::Float64, σ::Float64)::Float64
        a.*exp(.-(x-μ).^2 ./ (2 .* σ.^2))
    end

    """
        gaussian_2d(x::Float64, y::Float64, a::Float64, μx::Float64, μy::Float64, σx::Float64, σy::Float64)::Float64

    Compute the 2D gaussian at point x,y for an amplitude a, mean value μx and μy,
    and a standard deviation σx and σy
    """
    function gaussian_2d(x::Float64, y::Float64, a::Float64, μx::Float64, μy::Float64, σx::Float64, σy::Float64)::Float64
        a.*exp(.-((x.-μx).^2 ./ (2 .* σx.*σx) + (y.-μy).^2 ./ (2 .* σy.*σy)))
    end

    """
        gaussian_2d_rotated(x::Float64, y::Float64, amplitude::Float64, μx::Float64, μy::Float64, σx::Float64, σy::Float64, θ::Float64)::Float64

    Compute the rotated 2D gaussian at point x,y for an amplitude a, mean value μx and μy, standard deviation σx and σy
    and rotation angle θ
    """
    function gaussian_2d_rotated(x::Float64, y::Float64, amplitude::Float64, μx::Float64, μy::Float64, σx::Float64, σy::Float64, θ::Float64)::Float64
        a = cos(θ)^2 / (2*σx^2) + sin(θ)^2 / (2*σy^2)
        b = sin(2*θ) / (2*σx^2) - sin(2*θ) / (2*σy^2)
        c = sin(θ)^2 / (2*σx^2) + cos(θ)^2 / (2*σy^2)

        amplitude.*exp(- a*(x-μx)^2 - b*(x-μx)*(y-μy) - c*(y-μy)^2)
    end

    """
        multipass(x::Float64, y::Float64, amplitudes::Vector{Float64}, xlocs::Vector{Float64}, ylocs::Vector{Float64}, σx::Float64, σy::Float64)::Float64

    generate a multipass with 2D gaussian profiles for each pass
    """
    function multipass_2d(x::Float64, y::Float64, amplitudes::Vector{Float64}, xlocs::Vector{Float64}, ylocs::Vector{Float64}, σx::Float64, σy::Float64)::Float64
        intensity::Float64 = 0.0
        for i = 1:length(amplitudes::Vector{Float64})
            @inbounds intensity += gaussian_2d(x,y,amplitudes[i],xlocs[i],ylocs[i], σx,σy)
        end
        return intensity
    end
end