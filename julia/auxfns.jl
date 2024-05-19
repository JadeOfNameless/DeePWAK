import Base.map
import DataFrames.subset
import DataFrames.select

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

function clusts(C::AbstractMatrix)
    return map(x->x[1],argmax(C,dims=1))
end

function unhot(x)
    map(i->i[1],argmax(x,dims=1)) .- 1

end

function zerodiag(G::AbstractArray)
    m, n = size(G)
    G = G .* (1 .- I(n))
    return G
end

# [CuArray] -> [CuArray]
# workaround for moving identity matrix to GPU 
function zerodiag(G::CuArray)
    m, n = size(G)
    G = G .* (1 .- I(n) |> gpu)
    return G
end

function neighborcutoff(G::AbstractArray; ϵ=0.0001)
    M = G .> ϵ
    return G .* M
end

function not0(X)
    return filter(x->x != 0,X)
end

function rep(expr,n)
    return map(_->expr,1:n)
end

function map(f::Function,dict::Dict)
    args = zip(keys(dict),values(dict))
    res = map(args) do (lab,val)
        (lab,f(val))
    end
    return Dict(res)
end

function maplab(f::Function,dict::Dict)
    args = zip(keys(dict),values(dict))
    g = (key,val)->(key,f(key,val))
    res =  map(x->g(x...),args)
    return Dict(res)
end

function repkeys(dict::Dict,dim=1)
    sp = maplab(dict) do (lab,M)
        m = size(M)[dim]
        return rep(lab,m)
    end
    return sp
end

function repkey_clust(dict::Dict)
    sp = maplab(dict) do (lab,M)
        m = (length ∘ unique)(M)
        return rep(lab,m)
    end
    return sp
end

function dictcat(l)
    sel = (collect ∘ union)(map(keys,l)...)
    res = map(sel) do key
        vals = map(dict->dict[key],l)
        vals = map(x->hcat(x...),
                eachcol(map(x->vcat(x...),
                            eachrow(vals))))
        return (key,vals[1])
    end
    return Dict(res)
end

function mapkey(f,dicts::Dict...)
    sel = (collect ∘ intersect)(map(keys,dicts)...)
    res = map(sel) do i
        args = map(D->D[i],dicts)
        res = f(args...)
        return (i,res)
    end
    
    return Dict(res)
end

function subset(dict::Dict,keys::AbstractArray)
    Dict(key => dict[key] for key ∈ keys if haskey(dict,key))
end

function select(dict::Dict,keys::AbstractArray)
    return map(i->dict[i],keys)
end
             
