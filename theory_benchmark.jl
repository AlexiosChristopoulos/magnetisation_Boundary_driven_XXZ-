using BoundaryDrivenXXZ, LinearAlgebra, Printf, Test
BLAS.set_num_threads(1)
mkpath(joinpath(@__DIR__,"results"))
# At delta=0 the Jordan-Wigner hopping is 2J, edge relaxation is 2gamma,
# and the spin current is twice the particle current. Thus jZ=8gamma J² mu/
# (gamma²+4J²), independent of chain length. This is a derived XX limit;
# it does not assert reproduction of Prosen's interacting maximally driven NESS.
@testset "XX stationary spin current and MPO convergence" begin
    open(joinpath(@__DIR__,"results","theory_current.csv"),"w") do io
        println(io,"N,gamma,mu,J,dense,theory,error,mpo_dt,mpo,mpo_error")
        for N in 2:4, gamma in (.2,.7,2.)
            J=.5;mu=.6;m=Model(N;J,delta=0,gamma,mu)
            rho=dense_steady_state(m)
            # Reservoir balance supplies a spin current independent of the bond estimator.
            z1=reduce(kron,[k==1 ? BoundaryDrivenXXZ.Z : BoundaryDrivenXXZ.I2 for k in N:-1:1])
            dense=2gamma*(mu-real(tr(z1*rho)))
            theory=8gamma*J^2*mu/(gamma^2+4J^2)
            @test dense ≈ theory atol=2e-12
            if N==3 && gamma==.7
                previous_error=Inf
                for dt in (.1,.05,.025)
                    obs=observables(m,evolve(m,30.;dt,cutoff=1e-14,maxdim=64))
                    err=maximum(abs.(obs.current .- theory))
                    @test err < .002
                    @test err < previous_error/3.5
                    previous_error=err
                    println(io,join((N,gamma,mu,J,dense,theory,abs(dense-theory),dt,obs.current[1],err),","))
                end
            else
                println(io,join((N,gamma,mu,J,dense,theory,abs(dense-theory),NaN,NaN,NaN),","))
            end
        end
    end
end
