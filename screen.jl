include("prelude.jl")

using ThreadTools

ks = [3:30...]
function mlp(l::AbstractVector{<: Integer},f::Function)
    θ = foldl(l[3:length(l)],
              init=Chain(Dense(l[1] => l[2],f))) do layers,d
        d_0 = size(layers[length(layers)].weight)[1]
        return Chain(layers...,Dense(d_0 => d,f))
    end
end

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
    L_cl = 
end
    
function nearestneighbor(D)
    rows, cols = size(D)

    # Output matrix initialized to zeros on the GPU
    result = CUDA.zeros(Float32, rows, cols)

    # Define a CUDA kernel to mark the argmax in each column with 1
    function mark_argmax_kernel(D, result)
        col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if col <= cols
            max_index = 1
            max_value = D[1, col]
            for row in 1:rows
                if D[row, col] > max_value
                    max_value = D[row, col]
                    max_index = row
                end
            end
            result[max_index, col] = 1.0
        end
        return
    end

    # Launch the kernel with enough blocks to cover all columns
    @cuda threads=256 blocks=(cols + 255) ÷ 256 mark_argmax_kernel(D, result)

    return result
end
using CUDA

function gpu_knn(D::CuArray{Float32, 2}, k::Int)
    rows, cols = size(D)

    # Output matrix initialized to zeros on the GPU
    result = CUDA.zeros(Float32, rows, cols)

    # Additional matrix to keep track of the values that have been taken
    taken = CUDA.zeros(Bool, rows, cols)

    # Define a CUDA kernel to mark the top k values in each column with 1
    function mark_top_k_kernel(D, result, taken, k)
        col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
        if col <= cols
            for _ in 1:k
                max_index = 0
                max_value = -Inf
                for row in 1:rows
                    if !taken[row, col] && D[row, col] > max_value
                        max_value = D[row, col]
                        max_index = row
                    end
                end
                if max_index != 0
                    result[max_index, col] = 1.0
                    taken[max_index, col] = true
                end
            end
        end
        return
    end

    # Launch the kernel with enough blocks to cover all columns
    @cuda threads=256 blocks=(cols + 255) ÷ 256 mark_top_k_kernel(D, result, taken, k)

    return result
end


function knn(D,k)
    if k == 0
        return 0
    else
        nn = nearestneighbor(D)
        mask = 1 .- nn
        return nn .+ knn(D .* mask,k -1)
    end
end


function train_k!(M,path::String,
                loader::Flux.DataLoader,
                opt::Flux.Optimiser,
                epochs::Integer;
                savecheckpts=true)
    if length(path) > 0
        mkpath(path)
    end
    log = []
    log_k = []
    @showprogress map(1:epochs) do i
        E_X = encode(M,X)
        D_X = dist(M,E_X)
        L_k = map(ks) do k
            G = gpu_knn(D_X,k)
            Ê = (wak(D_X .* G) * E_X')'
            X̂ = decode(M_eucl,Ê)
            return Flux.mse(X̂,X)
        end
        push!(log_k,[L_k...]')
        k = ks[argmin(L_k)]
        map(loader) do (x,y)
            f = m->begin
                L = loss_k(m,x,k)
                L_test = loss_k(m,test,k)
                return L,L_test
            end
            l = TrainingIO.update!(M,f,opt)
            push!(log,[l...]')
        end
        if savecheckpts
            savemodel(M,path*"/"*string(i))
        end
    end
    log = vcat(log...)
    if length(path) > 0
        savemodel(M,path*"/final")
        #Tables.table(vcat(log...)) |> CSV.write(path*"/loss.csv")
        writecsv(log,path*"/loss.csv")
        writecsv(log_k,path*"/loss_k.csv")
#        plotloss(log[:,1],"loss",path*"/loss.pdf")
    end
    return log
end


groups = (DataFrame ∘ CSV.File)("data/groups.csv",normalizenames=true)

dat = (DataFrame ∘ CSV.File)("data/z_dat.csv",normalizenames=true);
dat = Matrix(dat[:,2:end]);
dat = hcat(filter(x->sum(x) != 0,eachslice(dat,dims=2))...);

X = scaledat(dat')
m,n = size(X)
l = 5

frac = 10
n_test = Integer(2^round(log2(n) - log2(frac)))
n_train = n - n_test
test,train = sampledat(X,n_test) |> gpu
X = gpu(X)

loader = Flux.DataLoader((train,train),batchsize=n_test,shuffle=true) 

layers = accumulate(÷,rep(2,4),init=2*m)
d = last(layers)

θ = mlp(layers,tanh)
ϕ = mlp(reverse(layers),tanh)

M_eucl = DistEnc(θ,ϕ,inveucl) |> gpu

L = train!(M_eucl,path*"/L2",(M,x,y)->Flux.mse(M(x),x),loader,opt,epochs)
L = train!(M_eucl,path*"L2/L1_L2",loss_cossim,loader,opt,epochs)

M_eucl = DistEnc(θ,ϕ,inveucl) |> gpu
L = train!(M_eucl,path*"L0_L1_L2",loss_cossim,loader,opt,epochs)

_,L,_ = load!(M_eucl,"data/2024-04-24/L2")

include("plotfns.jl")
plotloss(L,[""],"MSE","data/2024-04-24/L2/","loss.pdf")

train_k!(M_eucl,"data/2024-04-24/L2/L2_k",loader,opt,100)

E = encode(M_eucl,X)
D = (zerodiag ∘ dist)(M_eucl,E)
L_k = map(𝐤[1]:𝐤[2]) do k
    G = gpu_knn(D,k)
    Ê = (wak(D .* G) * E')'
    X̂ = decode(M_eucl,Ê)
    return Flux.mse(X̂,X)
end
