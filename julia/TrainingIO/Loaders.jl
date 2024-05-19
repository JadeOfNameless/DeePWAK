module Preprocessing
export sampledat,scaledat,unhot

using Distributions,InvertedIndices
# ∀ A:Type m,n:Int -> [A m n] -> k:Int -> ([A m k],[A m n-k])
function sampledat(X::AbstractArray,k)
    _,n = size(X)
    sel = sample(1:n,k,replace=false)
    test = X[:,sel]
    train = X[:,Not(sel)]
    return test,train
end

# ∀ m,n:Int -> [Float m n] -> [Float m n]
#scales each column (default) or row to [-1,1]
function scaledat(X::AbstractArray,dims=1)
    Y = X ./ maximum(abs.(X),dims=dims)
    Y[isnan.(Y)] .= 0
    return Y
end

function unhot(x)
    map(i->i[1],argmax(x,dims=1)) .- 1
end

end

module Loaders
export mnistloader

using Flux,MLDatasets,OneHotArrays

function mnistloader(batchsize::Integer)
    dat = MNIST(split=:train)[:]
    target = onehotbatch(dat.targets,0:9)

    m_x,m_y,n = size(dat.features)
    X = reshape(dat.features[:, :, :], m_x, m_y, 1, n)

    loader = Flux.DataLoader((X,target),
                            batchsize=batchsize,
                            shuffle=true)
    return loader
end

function loader(dat::DataType,
                batchsize::Integer)
    X = dat(split=:train)[:]
    target = onehotbatch(X.targets,range(extrema(X.targets)...))
    loader = Flux.DataLoader((X,target),
                             batchsize=batchsize,
                             shuffle=true)
    return loader
end
end
