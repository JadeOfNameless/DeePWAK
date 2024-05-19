module DictMap
export rep,map,maplab,repkeys,repkey_clust,dictcat,mapkey,subset,select,apply

import Base.map
import Base.mapreduce
import DataFrames.subset
import DataFrames.select

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
             
function apply(f,X,args...;kwargs...)
    return map(x->f(x,args...,kwargs...),X)
end
               
end
