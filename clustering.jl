using SparseArrays

function perm(D)
    n,_ = size(D)
    K = sortperm(D,dims=1,rev=true) .% n
    K[K .== 0] .= n
    return K
end

function knn(N,k)
    n,_ = size(N)
    N = N[1:k,:]
    J = reshape(N,:)
    I = mapreduce(i->rep(i,k),vcat,1:n)
    return sparse(I,J,1,n,n)
end
                  
function clusts(N,k,γ)
    K = knn(N,k)
    C = Leiden.leiden(K,"mod++",γ=γ)
    return C
end

function partitionmat(C)
    (sum ∘ map)(1:maximum(C)) do c
        x = C .== c
        return x * x'
    end
end


function loss_clust(X,F,D,N,k,γ,s,M_sparse,M_dense)
    C = clusts(N,k,γ)
    P = partitionmat(C) |> gpu
    G = wak(D .* P)
    F̂ = ((G^s) * F')'
    Ê = M_sparse.decoder(F̂)
    X̂ = decode(M_dense,Ê)
    return Flux.mse(X̂,X)
end

    
