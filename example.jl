using BoundaryDrivenXXZ, LinearAlgebra, Printf

m = Model(4; J=0.5, delta=0.3, gamma=0.5, mu=0.5)
for t in (0.0, 0.4, 2.0, 8.0)
    state = evolve(m,t; dt=0.05, maxdim=64)
    obs = observables(m,state)
    @printf("t=%4.1f trace=%.9f  z=%s  j=%s\n", t, real(obs.trace),
            string(round.(obs.magnetization;digits=6)),
            string(round.(obs.current;digits=6)))
    if t == 0.4
        @printf("  max density error against dense solver: %.3e\n",
                maximum(abs.(density_matrix(m,state)-dense_evolve(m,t))))
    end
end

small = Model(3)
steady = dense_steady_state(small)
state = evolve(small,12.0;dt=0.05,maxdim=16)
obs = observables(small,state)
@printf("N=3 late-time current = %s; edge sources = %.6f, %.6f\n",
        string(round.(obs.current;digits=6)),obs.left_source,obs.right_source)
@printf("N=3 max NESS density error = %.3e\n",
        maximum(abs.(density_matrix(small,state)-steady)))
