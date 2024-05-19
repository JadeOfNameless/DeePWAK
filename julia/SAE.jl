struct SAE
    weight::AbstractArray
    bias::AbstractArray
    σ::Function
end
@functor SAE 

function SAE(m,d,σ=relu)
    weight = randn(d,m)
    bias = randn(d)
    return SAE(weight,bias,σ)
end

function encode(M::SAE,x)
    return M.σ(M.weight * x .+ M.bias)
end

function decode(M::SAE,c)
    return M.weight' * c
end

function (M::SAE)(x)
    c = encode(M,x)
    x̂ =  decode(M,c)
    return x̂
end

struct PSAE
    sae
    partitioner
end
@functor PSAE

function encode(M::PSAE,X)
    return encode(M.sae,X)
end

function cluster(M::PSAE,X)
    return M.partitioner(X)
end

function partition(M::PSAE,X)
    return (pwak ∘ cluster)(M,X)
end

function encodepred(M::PSAE,X)
    E = encode(M,X)
    P = partition(M,X)
    return (P * E')'
end

function decode(M::PSAE,X)
    return decode(M.sae,X)
end


function (M::PSAE)(X::AbstractMatrix)
    Ehat = encodepred(M,X)
    return decode(M,Ehat)
end

    
function L1(M::SAE,α,x)
    c = encode(M,x)
    return α * sum(abs(c))
end

function L1(M::PSAE,α,x)
    c = encodepred(M,x)
    return α * sum(abs.(c))
end

function L2(M::Union{SAE,PSAE},lossfn,x,y)
    return lossfn(M(x),y)
end

function loss_SAE(M::SAE,α,lossfn,x,y)
    x = gpu(x)
    y = gpu(y)
    return L1(M,α,x) + L2(M,lossfn,x,y)
end

