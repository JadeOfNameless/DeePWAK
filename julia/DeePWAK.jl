using Flux, Functors, ProgressMeter

struct DeePWAK
    encoder::Chain
    partitioner::Chain
    decoder::Chain

end
@functor DeePWAK

function DeePWAK(l::AbstractVector,f::Function)
    θ = mlp(l,f)
    π = Chain(mlp(l,f),softmax)
    ϕ = mlp(reverse(l),f)
    return DeePWAK(θ, π, ϕ)
end

function DeePWAK(l_e::AbstractVector,
                 l_p::AbstractVector,
                 l_d::AbstractVector,f::Function)
    θ = mlp(l_e,f)
    π = Chain(mlp(l_p,f),softmax)
    ϕ = mlp(l_d,f)
    return DeePWAK(θ, π, ϕ)
end

function DeePWAK(l_e::AbstractVector,
                 l_p::AbstractVector,
                 l_d::AbstractVector,
                 f_e::Function, f_p::Function, f_d::Function)
    θ = mlp(l_e,f_e)
    π = Chain(mlp(l_p,f_p),softmax)
    ϕ = mlp(l_d,f_d)
    return DeePWAK(θ, π, ϕ)
end

function DeePWAK(m::Integer,d::Integer,c::Integer,l::Integer,
                 f_e::Function,f_p::Function)
    θ = mlp4x(m,d,l,f_e)
    π = Chain(mlp4x(m,c,l,f_p),softmax)
    ϕ = mlp4x(d,m,l,f_e)
    return DeePWAK(θ, π, ϕ)
end

function DeePWAK(l::Integer,s::Integer,
                 m::Integer,d::Integer,c::Integer,
                 f_e::Function,f_p::Function)
    θ = mlp(m,d,l,s,f_e)
    ϕ = mlp(d,m,l,s,f_e)
    π = Chain(mlp(m,c,l,s,f_p),softmax)
    return DeePWAK(θ,π,ϕ)
end

# ∀ n:Int ∃ m:Int -> [Float m n] -> [Float m n]
function (M::DeePWAK)(X::AbstractMatrix)
    #w = (softmax ∘ m.weigher)(X)
    #E = w .* m.encoder(X)
    E = M.encoder(X)
    C = M.partitioner(X)
    P = C' * C
    G = wak(P)
    Ehat = (G * E')'
    return M.decoder(Ehat)
end

function update!(M,loss::Function,opt)
    state = Flux.setup(opt,M)
    l,∇ = Flux.withgradient(loss,M)
    Flux.update!(state,M,∇[1])
    return l
end

function train!(M,loader,opt,test::AbstractMatrix,epochs::Integer)
    L = @showprogress map(1:epochs) do _
        map(loader) do (x,y)
            l_test = Flux.mse(M(test),test)
            l_train = update!(M,f->Flux.mse(f(x),x),opt)
            return l_train,l_test
        end
    end
    return L
end

function train!(M,loader,opt,epochs::Integer)
    L = @showprogress map(1:epochs) do _
        map(loader) do (x,y)
            l = update!(M,f->Flux.mse(f(x),x),opt)
            return l
        end
    end
    return L
end

function train!(M,loader,opt,test::AbstractMatrix,epochs::Integer,
                loss::Function)
    L = @showprogress map(1:epochs) do _
        map(loader) do (x,y)
            l_test = Flux.mse(M(test),test)
            l_train = update!(M,m->loss(m,x),opt)
            return l_train,l_test
        end
    end
    return L
end

function train!(M,loader,opt,epochs::Integer,loss::Function)
    L = @showprogress map(1:epochs) do _
        map(loader) do (x,y)
            l = update!(M,m->loss(m,x),opt)
            return l
        end
    end
    return L
end

struct DeePWAKBlock
    block::Parallel
end
@functor DeePWAKBlock

function DeePWAKBlock(h::Integer,l::Integer,s::Integer,
                      m::Integer,d::Integer,c::Integer,
                      f_e::Function,f_p::Function; combine=vcat)
    block = map(1:h) do _
        return DeePWAK(l,s,m,d,c,f_e,f_p)
    end
    return (DeePWAKBlock ∘ Parallel)(combine,block...)
end

function DeePWAKBlock(h::Integer,l::Integer,s::Integer,
                      m::Integer,d::Integer,c::Integer,
                      f_e::Function,f_p::Function,f_d::Function;
                      combine=vcat)
    block = map(1:h) do _
        return DeePWAK(l,s,m,d,c,f_e,f_p,f_d)
    end
    return (DeePWAKBlock ∘ Parallel)(combine,block...)
end

function (M::DeePWAKBlock)(X::AbstractMatrix)
    return M.block(X)
end

struct BlockList
    first
    rest
end

function(B::BlockList)(X::AbstractMatrix)
    B.rest(hcat(X,B.first(X)))
end

    

    
