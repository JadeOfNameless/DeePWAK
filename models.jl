function mlp(l::AbstractVector{<: Integer},f::Function)
    θ = foldl(l[3:length(l)],
              init=Chain(Dense(l[1] => l[2],f))) do layers,d
        d_0 = size(layers[length(layers)].weight)[1]
        return Chain(layers...,Dense(d_0 => d,f))
    end
end

function loss_cossim(α::AbstractFloat,f,M,x::AbstractArray,y::AbstractArray)
    E = encode(M,x)
    C = cossim(E')
    L1 = α * sum(abs.(C))
    ŷ = decode(M,E)
    L2 = f(ŷ,y)
    return L1+L2,L1,L2
end

function loss_test(f::Function,M,test::AbstractArray,train::AbstractArray,_)
    L_train = f(M,train,train)
    L_test = f(M,test,test)
    return (L_train...,L_test...)
end
function loss_test(f::Function, M, test::AbstractArray, train::AbstractArray)
    # Apply the function to the training data and convert the result to a tuple if not already
    train_result = f(M, train, train)
    train_tuple = typeof(train_result) <: Tuple ? train_result : (train_result,)

    # Apply the function to the test data and convert the result to a tuple if not already
    test_result = f(M, test, test)
    test_tuple = typeof(test_result) <: Tuple ? test_result : (test_result,)

    # Concatenate the tuples from training and test results
    return (train_tuple..., test_tuple...)
end

function loss_dense(α::AbstractFloat,M::Autoencoder,
                    test::AbstractArray,train::AbstractArray;f=Flux.mse)
    return loss_test((ϕ,x,_)->loss_cossim(α,f,ϕ,x,x),M,test,train)
end

function sparsecov(E::AbstractArray)
    n = sum(abs.(E))
    return abs.(cossim(E')) * (sum(abs.(E),dims=2) ./ n) * (sum(abs.(E),dims=1) ./ n) * sindiff(E)
end

function loss_sparsecov(α::AbstractFloat,M::Autoencoder,x,y;f=Flux.mse)
    E = encode(M,x)
    L1 = α * (sum ∘ sparsecov)(E)
    ŷ = decode(M,E)
    L2 = f(ŷ,y)
    return L1 + L2, L1, L2
end

function sparsitymat(α,β,E::AbstractArray)
    C = α .* cossim(E')
    D = β .* sindiff(E)
    return C * E * D
end

function sparsitymat(α,β,M::SparseEncoder,x::AbstractArray)
    E = encode(M,x)
    return sparsitymat(α,β,E)
end

function l2_dense(M::Autoencoder,test::AbstractArray,train::AbstractArray)
    return loss_test((ϕ,x,_)->Flux.mse(ϕ(x),x),M,test,train)
end

function l2_dense(M::SparseEncoder,test::AbstractArray,train::AbstractArray)
    return loss_test((ϕ,x,_)->Flux.mse(ϕ(x),x),M,test,train)
end

function l2_sparse(M_sparse::SparseEncoder,M_dense::Autoencoder,test,train)
    g = (θ,x,_)->lossfn_sparse(Flux.mse,M_dense)((θ ∘ encode)(M_dense,x),x)
    return loss_test(g,M_sparse,test,train)
end

function loss_sindiff(α::AbstractFloat,f::Function,M,x::AbstractArray,y::AbstractArray)
    E = encode(M,x)
    D = sindiff(E)
    L1 = α * sum(D * E')
    ŷ = decode(M,E)
    L2 = f(ŷ,y)
    return L1+L2,L1,L2
end

function lossfn_sparse(f::Function,M_dense::Autoencoder)
    return (Ê,y)->f(decode(M_dense,Ê),y)
end
    
function loss_sparse(α::AbstractFloat,
                     M_sparse::SparseEncoder,M_dense::Autoencoder,
                     test::AbstractArray,train::AbstractArray;f=Flux.mse)
    g = (M,x,_)->begin
        E = encode(M_dense,x)
        return loss_sindiff(α,lossfn_sparse(f,M_dense),M,E,x)
    end
    return loss_test(g,M_sparse,test,train)
end

    
