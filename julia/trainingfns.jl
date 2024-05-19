using ProgressMeter

function update!(M,x,y,loss::Function,opt)
    x = gpu(x)
    y = gpu(y)
    f = m->loss(m(x),y)
    state = Flux.setup(opt,M)
    l,∇ = Flux.withgradient(f,M)
    Flux.update!(state,M,∇[1])
    return l
end

function update!(M,loss::Function,opt)
    state = Flux.setup(opt,M)
    l,∇ = Flux.withgradient(loss,M)
    Flux.update!(state,M,∇[1])
    return l
end

function train!(M,loader,opt,epochs,loss,log)
    @showprogress map(1:epochs) do _
         map(loader) do (x,y)
            l = update!(M,x,y,loss,opt)
            push!(log,l)
        end
    end
end

function train!(sae::Union{SAE,PSAE},M_outer,α,loader,opt,epochs,lossfn,log)
    @showprogress map(1:epochs) do _
        map(loader) do (x,y)
            f = loss_SAE(M_outer,α,lossfn,x)
            state = Flux.setup(opt,sae)
            l,∇ = Flux.withgradient(f,sae)
            Flux.update!(state,sae,∇[1])
            push!(log,l)
        end
    end
end

function outerenc(m=3)
    kern = (3,3)
    s = (2,2)
    θ_conv = Chain(Conv(kern,1 => 3,relu,stride=s),
                Conv(kern,3 => 6,relu,stride=s),
                Conv(kern,6 => 9,relu,stride=s),
                Conv((2,2),9 => 12,relu))

    θ_mlp = Chain(Dense(12 => 6,relu),
                Dense(6 => m,relu))

    θ_outer = Chain(θ_conv,
                    x->reshape(x,12,:),
                    θ_mlp)
    return θ_outer
end

function outerclassifier(m=3)
    π_outer = Chain(Dense(m => 5,relu),
                    Dense(5 => 10,relu),
                    softmax)
    return π_outer
end

function outerdec(m=3)
    kern = (3,3)
    s = (2,2)
    ϕ_mlp = Chain(Dense(m => 6,relu),
                  Dense(6 => 12,relu))
    ϕ_deconv = Chain(ConvTranspose(kern,12 => 9,relu,stride=s),
                   ConvTranspose(kern,9 => 6,relu,stride=s),
                   ConvTranspose(kern,6 => 3,relu,stride=s),
                   ConvTranspose(kern,3 => 1,relu,stride=s))
    ϕ = Chain(ϕ_mlp,ϕ_deconv)
    

function outermodel()
    kern = (3,3)
    s = (2,2)
    θ_conv = Chain(Conv(kern,1 => 3,relu,stride=s),
                Conv(kern,3 => 6,relu,stride=s),
                Conv(kern,6 => 9,relu,stride=s),
                Conv((2,2),9 => 12,relu))

    θ_mlp = Chain(Dense(12 => 6,relu),
                Dense(6 => m,relu))

    θ_outer = Chain(θ_conv,
                    x->reshape(x,12,:),
                    θ_mlp)

    π_outer = Chain(Dense(m => 5,relu),
                    Dense(5 => 10,relu),
                    softmax)

    M_outer = Chain(θ_outer,π_outer)
    return M_outer

end
