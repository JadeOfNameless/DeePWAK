using CUDA

function kern_euclidean!(result, X)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    for j = 1:size(result, 2)
        if i <= size(X, 1) && j <= size(X, 1)
            temp = 0.0f0
            for k = 1:size(X, 2)
                temp += (X[i, k] - X[j, k])^2
            end
            result[i, j] = sqrt(temp)
        end
    end
    return
end

function euclidean(X::CuArray{Float32},threads=1024)
    m, n = size(X)
    result = CUDA.zeros(Float32, m, m)
    threads_per_block = threads
    blocks = cld(m, threads_per_block)
    @cuda threads=threads_per_block blocks=blocks kern_euclidean!(result, X)
    CUDA.synchronize()
    return Array(result)
end

function euclidean(X::Adjoint{Float32, <:CuArray{Float32, 2}})
    # Extract the parent array and proceed with computation
    return euclidean(CuArray(X))
end

using CUDA
using Zygote

function pairwise_euclidean_distance(x::CuArray{Float32})
    m, n = size(x)
    x2 = sum(x .^ 2, dims=1)
    dist = x2' .+ x2 .- 2 * x' * x
    dist = sqrt.(max.(dist, 0))  # Numerical stability: possible small negative numbers due to precision errors
    return dist
end

# Now you can differentiate the function using Zygote
# Example with random data
using Random

Random.seed!(123)
data = CUDA.rand(Float32, 10, 100)  # 10-dimensional, 100 data points

distance = pairwise_euclidean_distance(data)
gradient = Zygote.gradient(x -> sum(pairwise_euclidean_distance(x)), data)
