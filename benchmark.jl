using BoundaryDrivenXXZ, Printf

# Compile once before timing so the measurement covers solver execution.
for n in (4,6)
    model = Model(n)
    observables(model,evolve(model,1.0;dt=0.05,maxdim=32))
    elapsed = @elapsed begin
        state = evolve(model,1.0;dt=0.05,maxdim=32)
        global result = observables(model,state)
    end
    @printf("N=%d, t=1, dt=.05, maxdim=32: %.3f s; trace %.9f, current %.6f\n",
            n,elapsed,real(result.trace),result.current[1])
end
