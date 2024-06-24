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
end