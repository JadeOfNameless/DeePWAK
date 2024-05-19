module SaveDate
export date,savepath,savedate

using Dates,Plots

function date() :: String
    Dates.format(today(),"yyyy-mm-dd")*"/"
end

function savepath(path::String,file::String,args...;kwargs...) :: Nothing
    mkpath(path)
    save(path*"/"*file,args...;kwargs...)
end

function savepath(p::Plots.Plot,file::String)
    savefig(p,file)
end

function savepath(p::Plots.Plot,path::String,file::String)
    mkpath(path)
    savepath(p,path*"/"*file)
end

function savedate(path::String,args...;kwargs...) :: Nothing
    savepath(date()*"/"*path,args...;kwargs...)
end
end
