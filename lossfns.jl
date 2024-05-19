function loss_cossim(M,x,_)
    Ê = diffuse(M,x)
    x̂ = decode(M,Ê)
    
    L_0 = β * sum(abs.(cossim(Ê')))
    L_1 = α * sum(abs.(Ê))
    L_2 = Flux.mse(x̂,x)
    return L_0 + L_1 + L_2,L_0,L_1,L_2
end

function loss_test(M,x,_)
    L,L_0,L_1,L_2 = loss_cossim(M,x,x)
    L_test,L0_test,L1_test,L2_test = loss_cossim(M,test,test)
    return L,L_0,L_1,L_2,L_test,L0_test,L1_test,L2_test
end

function loss_k(M::SparseEncoder,x::CuArray,k::Integer)
    E_X = encode(M,X)
    D_X = dist(M,E_X)
    G = gpu_knn(D_X,k)
    Ê = (wak(D_X .* G) * E_X')'
    X̂ = decode(M_eucl,Ê)
    return Flux.mse(X̂,X)
end

function loss_k(M::SparseEncoder,x::CuArray)
    E_X = encode(M,X)
    D_X = dist(M,E_X)
    L_k = map(ks) do k
        G = gpu_knn(D_X,k)
        Ê = (wak(D_X .* G) * E_X')'
        X̂ = decode(M_eucl,Ê)
        return Flux.mse(X̂,X)
    end
    k = ks[argmin(L_k)]

    E_x = encode(M,x)
    D_x = dist(M,E_x)
    G = gpu_knn(D_x,k)
        
    Ê = (wak(D_x .* G) * E_x')'
    x̂ = decode(M_eucl,Ê)
    return Flux.mse(x̂,x),L_k
end

function loss(M,x,_)
    L = loss_cossim(M,x,x)
    L_test = loss_cossim(M,test,test)

    E = encode(M,X)
    D = dist(M,E)
    L_k = map(𝐤[1]:𝐤[2]) do k
        G = gpu_knn(D,k)
        Ê = (wak(D .* G) * E')'
        X̂ = decode(M_eucl,Ê)
        return Flux.mse(X̂,X)
    end
    k = argmin(L_k)
    G = gpu_knn(D,k)

    tmp=Leiden.leiden(sparse(cpu(G .* D)),"mod++")
    P = mapreduce(k->tmp .== k,hcat,tmp)
#    L_cl = 
end
    
