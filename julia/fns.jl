using Distributions, ThreadTools, Flux, InvertedIndices

function zcat(args...)
    cat(args...,dims=3)
end

function mapmap(f,args...)
    map((args...)->map(f,args...),args...)
end

function tmapmap(f,args...)
    tmap((args...)->map(f,args...),args...)
end

function tmapreduce(f,rf,args...)
    rf(tmap(f,args...)...)
end

function scaledat(X::AbstractArray,dims=1)
    Y = X ./ maximum(abs.(X),dims=dims)
    Y[isnan.(Y)] .= 0
    return Y
end

function sampledat(X::AbstractArray,k)
    _,n = size(X)
    sel = sample(1:n,k,replace=false)
    test = X[:,sel]
    train = X[:,Not(sel)]
    return test,train
end

function wak(G)
    #Matrix -> Matrix
    #returns G with row sums normalized to 1
    W = sum(G,dims=2)
    K = G ./ W
    K[isnan.(K)] .= 0
    return K
end

function ehat(E,D,G)
    (wak(G .* D) * E')'
end

function 𝕃(X,θ,E,D,G)
    Flux.mse(X,(θ ∘ ehat)(E,D,G))
end

function partitionmat(C)
    (sum ∘ map)(1:maximum(C)) do c
        x = C .== c
        return x * x'
    end
end

function diffuse(X,θ,E,D,G,P,s)
    M = P .* G
    M = wak(M .* D)
    foldl(1:s,init=(M,[])) do (M⁺,L),_
        M⁺ = M⁺ * M
        L⁺ = Flux.mse(X,θ((M⁺ * E')'))
        L = vcat(L,L⁺)
        return M⁺,L
    end
end


function embedding(Θ,X)
    map(Θ.layers) do θ
        θ[1](X)|>cpu
    end
end

function distmat(E)
    1 ./ (pairwise(Euclidean(),E,E) + (I*Inf));
end

function perm(D,n)
    K = sortperm(D,dims=1,rev=true) .% n
    K[K .== 0] .= n
    return K
end

function adjmat(K,𝐤,n)
    G = map(1:maximum(𝐤)) do k
        sparse(1:n,K[:,k],1,n,n)
    end
    G = map(𝐤) do k
        foldl(+,G[1:k])
    end
    return G
end
        
function 𝕃_dk(X,θ,E,D,𝐆)
    E = E |> gpu
    D = D |> gpu
    map(𝐆) do G
        G = G |> gpu
        return 𝕃(X,θ[2],E,D,G)
    end
end
