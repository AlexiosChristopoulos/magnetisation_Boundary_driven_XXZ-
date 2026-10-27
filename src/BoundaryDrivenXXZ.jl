module BoundaryDrivenXXZ

using ITensors, ITensorMPS, LinearAlgebra

export Model, mixed_state, evolve, observables, density_matrix, density_mpo, dense_evolve,
       dense_steady_state

const I2 = Matrix{ComplexF64}(I, 2, 2)
const X = ComplexF64[0 1; 1 0]
const Y = ComplexF64[0 -im; im 0]
const Z = ComplexF64[1 0; 0 -1]
const PLUS = ComplexF64[0 1; 0 0]
const MINUS = PLUS'

"""XXZ chain with H=J Σ(XX+YY)+Δ ΣZZ and edge polarization ±μ.

The four edge jumps have rates γ(1±μ). The local MPDO index orders
`(ket,bra)` as `(↑↑,↓↑,↑↓,↓↓)` (Julia column major order).
"""
struct Model
    sites::Vector{Index}
    J::Float64
    delta::Float64
    gamma::Float64
    mu::Float64
end

function Model(n::Integer; J=0.5, delta=0.3, gamma=0.5, mu=0.5)
    n >= 2 || throw(ArgumentError("the chain requires at least two sites"))
    all(x -> x isa Real && isfinite(x), (J, delta, gamma, mu)) ||
        throw(ArgumentError("parameters must be finite real numbers"))
    gamma >= 0 || throw(ArgumentError("gamma must be nonnegative"))
    abs(mu) <= 1 || throw(ArgumentError("abs(mu) must be at most one"))
    sites = [Index(4, "Site,n=$i") for i in 1:n]
    return Model(sites, Float64(J), Float64(delta), Float64(gamma), Float64(mu))
end

nbonds(m::Model) = length(m.sites)-1
hbond(m::Model) = m.J*(kron(X,X)+kron(Y,Y)) + m.delta*kron(Z,Z)

# A basis state of a two-site density operator, ordered by local Liouville
# indices q_i + 4(q_j-1). Tensor factors in kron use site j then site i.
function pair_matrix(qi, qj)
    ai, bi = mod1(qi,2), (qi-1)÷2+1
    aj, bj = mod1(qj,2), (qj-1)÷2+1
    rho = zeros(ComplexF64, 4, 4)
    rho[ai+2*(aj-1), bi+2*(bj-1)] = 1
    return rho
end

function pair_vector(rho)
    out = zeros(ComplexF64, 16)
    for qj in 1:4, qi in 1:4
        ai, bi = mod1(qi,2), (qi-1)÷2+1
        aj, bj = mod1(qj,2), (qj-1)÷2+1
        out[qi+4*(qj-1)] = rho[ai+2*(aj-1), bi+2*(bj-1)]
    end
    return out
end

dissipator(rho, l) = l*rho*l' - (l'*l*rho + rho*l'*l)/2

function edge_jumps(m::Model, bond::Int)
    jumps = Matrix{ComplexF64}[]
    if bond == 1
        push!(jumps, sqrt(m.gamma*(1+m.mu))*kron(I2, PLUS))
        push!(jumps, sqrt(m.gamma*(1-m.mu))*kron(I2, MINUS))
    end
    if bond == nbonds(m)
        push!(jumps, sqrt(m.gamma*(1-m.mu))*kron(PLUS, I2))
        push!(jumps, sqrt(m.gamma*(1+m.mu))*kron(MINUS, I2))
    end
    return jumps
end

function bond_generator(m::Model, bond::Int)
    h = hbond(m)
    jumps = edge_jumps(m, bond)
    l = zeros(ComplexF64, 16, 16)
    for qj in 1:4, qi in 1:4
        rho = pair_matrix(qi, qj)
        drho = -im*(h*rho-rho*h)
        for jump in jumps
            drho += dissipator(rho, jump)
        end
        l[:, qi+4*(qj-1)] = pair_vector(drho)
    end
    return l
end

function gate(m::Model, bond::Int, step)
    a, b = m.sites[bond], m.sites[bond+1]
    matrix = exp(step*bond_generator(m, bond))
    tensor = ITensor(prime(a), prime(b), a, b)
    for qj in 1:4, qi in 1:4, pj in 1:4, pi in 1:4
        tensor[prime(a)=>pi, prime(b)=>pj, a=>qi, b=>qj] =
            matrix[pi+4*(pj-1), qi+4*(qj-1)]
    end
    return tensor
end

"""Identity density matrix divided by 2^N, as an MPDO in fused local indices."""
function mixed_state(m::Model)
    tensors = ITensor[]
    for s in m.sites
        tensor = ITensor(s)
        tensor[s=>1] = 0.5
        tensor[s=>4] = 0.5
        push!(tensors, tensor)
    end
    return MPS(tensors)
end

function checked_controls(time, dt, cutoff, maxdim)
    time isa Real && isfinite(time) && time >= 0 || throw(ArgumentError("time must be finite and nonnegative"))
    dt isa Real && isfinite(dt) && dt > 0 || throw(ArgumentError("dt must be finite and positive"))
    cutoff isa Real && isfinite(cutoff) && cutoff >= 0 || throw(ArgumentError("cutoff must be nonnegative"))
    maxdim isa Integer && maxdim >= 1 || throw(ArgumentError("maxdim must be positive"))
    return Float64(time), Float64(dt)
end

"""Second-order local-Liouvillian evolution of the MPDO.

The result is an MPS over fused ket/bra site indices, mathematically an MPO.
Truncation is controlled by `cutoff` and `maxdim`; the scheme does not enforce
positivity after compression. Refine all three controls for numerical checks.
"""
function evolve(m::Model, time; dt=0.05, cutoff=1e-12, maxdim=64,
                initial=mixed_state(m))
    t, bound = checked_controls(time, dt, cutoff, maxdim)
    length(initial) == length(m.sites) && all(siteind(initial,i) == m.sites[i]
        for i in eachindex(m.sites)) || throw(ArgumentError("initial has incompatible sites"))
    steps = t == 0 ? 0 : ceil(Int, t/bound)
    steps == 0 && return deepcopy(initial)
    half_gates = [gate(m, i, t/(2steps)) for i in 1:nbonds(m)]
    state = deepcopy(initial)
    for _ in 1:steps
        state = apply(half_gates, state; cutoff, maxdim)
        state = apply(reverse(half_gates), state; cutoff, maxdim)
    end
    return state
end

local_weight(o) = vec(transpose(o))

function product_trace(state::MPS, operators::AbstractDict{Int,<:AbstractMatrix})
    scalar = ITensor(1.0)
    for i in 1:length(state)
        s = siteind(state, i)
        w = local_weight(get(operators, i, I2))
        v = ITensor(s)
        for q in 1:4
            v[s=>q] = w[q]
        end
        scalar *= state[i]*v
    end
    return scalar[]
end

"""Trace, site magnetizations, bond currents, and edge source terms.

Current orientation is left-to-right: `jᵢ=2J(XᵢYᵢ₊₁−YᵢXᵢ₊₁)`.
For unit trace, `d⟨Zᵢ⟩/dt = jᵢ₋₁−jᵢ` in the bulk.
"""
function observables(m::Model, state::MPS)
    z = product_trace(state, Dict{Int,Matrix{ComplexF64}}())
    mag = [real(product_trace(state, Dict(i=>Z))/z) for i in eachindex(m.sites)]
    currents = [real(2m.J*(product_trace(state, Dict(i=>X,i+1=>Y))-
                           product_trace(state, Dict(i=>Y,i+1=>X)))/z)
                for i in 1:nbonds(m)]
    return (trace=z, magnetization=mag, current=currents,
            left_source=2m.gamma*(m.mu-mag[1]),
            right_source=-2m.gamma*(m.mu+mag[end]))
end

"""Materialize the MPDO for small chains, with little-endian site ordering."""
function density_matrix(m::Model, state::MPS)
    n = length(m.sites)
    n <= 5 || throw(ArgumentError("density materialization is limited to five sites"))
    tensor = state[1]
    for i in 2:n
        tensor *= state[i]
    end
    data = Array(tensor, m.sites...)
    dim = 1 << n
    rho = zeros(ComplexF64, dim, dim)
    for bra in 0:dim-1, ket in 0:dim-1
        q = ntuple(i -> 1 + ((ket >> (i-1)) & 1) + 2*((bra >> (i-1)) & 1), n)
        rho[ket+1, bra+1] = data[q...]
    end
    return rho
end

"""Return the evolved MPDO as an ITensorMPS `MPO` with separate ket/bra legs."""
function density_mpo(m::Model, state::MPS)
    length(state) == length(m.sites) || throw(ArgumentError("state length mismatch"))
    tensors = ITensor[]
    for i in eachindex(m.sites)
        s = m.sites[i]
        siteind(state,i) == s || throw(ArgumentError("state site mismatch"))
        p = Index(2,"Site,n=$i")
        mapper = ITensor(s,prime(p),p)
        for bra in 1:2, ket in 1:2
            mapper[s=>ket+2*(bra-1),prime(p)=>ket,p=>bra] = 1
        end
        push!(tensors,state[i]*mapper)
    end
    return MPO(tensors)
end

function dense_generator(m::Model)
    n = length(m.sites)
    n <= 4 || throw(ArgumentError("dense solver is limited to four sites"))
    dim = 1 << n
    embedded(op, site) = reduce(kron, [k == site ? op : I2 for k in n:-1:1])
    x = [embedded(X,i) for i in 1:n]
    y = [embedded(Y,i) for i in 1:n]
    z = [embedded(Z,i) for i in 1:n]
    h = zeros(ComplexF64, dim, dim)
    for i in 1:n-1
        h += m.J*(x[i]*x[i+1]+y[i]*y[i+1]) + m.delta*z[i]*z[i+1]
    end
    unit = Matrix{ComplexF64}(I, dim, dim)
    l = -im*(kron(unit,h)-kron(transpose(h),unit))
    for (site, op, rate) in ((1,PLUS,m.gamma*(1+m.mu)),
                             (1,MINUS,m.gamma*(1-m.mu)),
                             (n,PLUS,m.gamma*(1-m.mu)),
                             (n,MINUS,m.gamma*(1+m.mu)))
        jump = sqrt(rate)*embedded(op,site)
        r = jump'*jump
        l += kron(conj(jump),jump) - (kron(unit,r)+kron(transpose(r),unit))/2
    end
    return l
end

"""Independent full-Hamiltonian dense Lindblad evolution from the mixed state."""
function dense_evolve(m::Model, time)
    t, _ = checked_controls(time, 1.0, 0.0, 1)
    dim = 1 << length(m.sites)
    return reshape(exp(t*dense_generator(m))*vec(Matrix{ComplexF64}(I,dim,dim)/dim),dim,dim)
end

"""Small-chain NESS from a trace-constrained dense linear solve."""
function dense_steady_state(m::Model)
    dim = 1 << length(m.sites)
    l = dense_generator(m)
    rhs = zeros(ComplexF64, dim^2)
    l[1,:] = vec(Matrix{ComplexF64}(I,dim,dim))
    rhs[1] = 1
    return reshape(l\rhs,dim,dim)
end

end
