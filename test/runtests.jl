using Test, BoundaryDrivenXXZ, ITensors, ITensorMPS, LinearAlgebra

const B = BoundaryDrivenXXZ

@testset "model and exact limits" begin
    @test_throws ArgumentError Model(1)
    @test_throws ArgumentError Model(3; gamma=-0.1)
    @test_throws ArgumentError Model(3; mu=1.1)
    @test_throws ArgumentError Model(3; J=Inf)
    m = Model(3; J=0, delta=0, gamma=0.7, mu=0.4)
    s = evolve(m, 0.6; dt=0.1)
    o = observables(m,s)
    expected = 0.4*(1-exp(-2*0.7*0.6))
    @test o.magnetization ≈ [expected, 0.0, -expected] atol=1e-11
    @test maximum(abs.(o.current)) < 1e-12
    @test abs(o.trace-1) < 1e-12
    @test_throws ArgumentError evolve(m, 1.0; dt=0)
end

@testset "user-supplied initial density" begin
    m = Model(3; J=0.5, delta=0.3, gamma=0.5, mu=0.5)
    # |up,down,up><up,down,up| in the fused local ket/bra basis.
    tensors = [ITensor(s) for s in m.sites]
    for (tensor, site, q) in zip(tensors, m.sites, (1, 4, 1))
        tensor[site=>q] = 1
    end
    initial = MPS(tensors)
    rho0 = density_matrix(m, initial)
    @test rho0[3, 3] == 1
    @test tr(rho0) == 1
    @test density_matrix(m, evolve(m, 0; initial=initial)) == rho0

    t = 0.2
    state = evolve(m, t; dt=0.01, cutoff=1e-14, maxdim=64, initial=initial)
    exact = reshape(exp(t*B.dense_generator(m))*vec(rho0), 8, 8)
    @test maximum(abs.(density_matrix(m, state)-exact)) < 3e-6
end

@testset "closed-chain unitary limit" begin
    m = Model(3; J=0.5, delta=0.3, gamma=0, mu=0.5)
    # The product density |up,down,up><up,down,up| must remain pure when
    # all boundary jumps vanish, despite coherent XXZ transport.
    tensors = [ITensor(s) for s in m.sites]
    for (tensor, site, q) in zip(tensors, m.sites, (1, 4, 1))
        tensor[site=>q] = 1
    end
    initial = MPS(tensors)
    t = 0.35
    state = evolve(m, t; dt=0.01, cutoff=1e-14, maxdim=64,
                   initial=initial)
    rho = density_matrix(m, state)
    rho0 = density_matrix(m, initial)
    exact = reshape(exp(t*B.dense_generator(m))*vec(rho0), 8, 8)
    @test maximum(abs.(rho-exact)) < 1e-5
    @test abs(tr(rho)-1) < 1e-10
    @test abs(real(tr(rho*rho))-1) < 1e-9
    @test minimum(eigvals(Hermitian((rho+rho')/2))) > -1e-9
    @test maximum(abs.(observables(m, state).current)) > 0.1
end

@testset "stationary and diagonal-interaction limits" begin
    # At zero bias every jump has an equally strong reverse jump, so I/2^N
    # is stationary even when exchange and the ZZ interaction are present.
    unbiased = Model(4; J=0.7, delta=0.9, gamma=0.6, mu=0)
    rho = density_matrix(unbiased, evolve(unbiased, 0.7; dt=0.07))
    @test rho ≈ Matrix{ComplexF64}(I, 16, 16)/16 atol=1e-11
    @test norm(B.dense_generator(unbiased)*vec(rho)) < 1e-10

    # With no exchange, the state remains diagonal. ZZ phases cannot alter
    # the independent edge relaxation from the maximally mixed state.
    diagonal = Model(4; J=0, delta=1.3, gamma=0.7, mu=0.4)
    t = 0.6
    state = evolve(diagonal, t; dt=0.06)
    obs = observables(diagonal, state)
    edge = diagonal.mu*(1-exp(-2diagonal.gamma*t))
    @test obs.magnetization ≈ [edge, 0.0, 0.0, -edge] atol=1e-11
    @test maximum(abs.(obs.current)) < 1e-12
    @test maximum(abs.(density_matrix(diagonal, state)-dense_evolve(diagonal, t))) < 1e-11
end

@testset "MPO physical-leg mapping" begin
    m = Model(2; J=0.5, delta=0.3, gamma=0.5, mu=0.5)
    state = evolve(m, 0.2; dt=0.02)
    mpo = density_mpo(m, state)
    tensor = mpo[1] * mpo[2]
    bra_indices = [only(filter(i -> hastags(i, "Site") && plev(i) == 0,
                               inds(mpo[k]))) for k in 1:2]
    from_mpo = reshape(Array(tensor, prime.(bra_indices)...,
                                  bra_indices...), 4, 4)
    @test from_mpo ≈ density_matrix(m, state) atol=1e-12
end

@testset "MPDO against independent dense Lindblad solver" begin
    for n in 2:4
        m = Model(n; J=0.5, delta=0.3, gamma=0.5, mu=0.5)
        s = evolve(m, 0.4; dt=0.02, cutoff=1e-14, maxdim=256)
        @test length(density_mpo(m,s)) == n
        rho = density_matrix(m,s)
        exact = dense_evolve(m,0.4)
        @test maximum(abs.(rho-exact)) < 3e-6
        @test abs(tr(rho)-1) < 1e-7
        @test minimum(eigvals(Hermitian((rho+rho')/2))) > -1e-8
        @test maximum(abs.(rho-rho')) < 1e-7
    end
end

@testset "continuity and convergence" begin
    m = Model(3; J=0.5, delta=0.3, gamma=0.5, mu=0.5)
    t = 0.3
    dt = 1e-5
    before = observables(m,evolve(m,t;dt=0.01))
    after = observables(m,evolve(m,t+dt;dt=0.01))
    derivative = (after.magnetization-before.magnetization)/dt
    rhs = [before.left_source-before.current[1],
           before.current[1]-before.current[2],
           before.current[2]+before.right_source]
    @test maximum(abs.(derivative-rhs)) < 2e-3
    exact = dense_evolve(m,0.5)
    coarse = density_matrix(m,evolve(m,0.5;dt=0.1,maxdim=16))
    fine = density_matrix(m,evolve(m,0.5;dt=0.025,maxdim=16))
    @test norm(fine-exact) < norm(coarse-exact)/4
    lowbond = density_matrix(m,evolve(m,0.5;dt=0.025,maxdim=2))
    @test norm(fine-exact) < norm(lowbond-exact)
end

@testset "small-chain NESS" begin
    m = Model(3; J=0.5, delta=0.3, gamma=0.5, mu=0.5)
    rho = dense_steady_state(m)
    @test abs(tr(rho)-1) < 1e-12
    @test norm(B.dense_generator(m)*vec(rho)) < 1e-12
    @test minimum(eigvals(Hermitian((rho+rho')/2))) > -1e-12
    s = evolve(m, 12.0; dt=0.05, cutoff=1e-13, maxdim=16)
    obs = observables(m,s)
    @test maximum(abs.(obs.current .- obs.current[1])) < 2e-4
    @test abs(obs.left_source-obs.current[1]) < 2e-4
    @test abs(obs.current[end]+obs.right_source) < 2e-4
    @test maximum(abs.(density_matrix(m,s)-rho)) < 2e-4
end
