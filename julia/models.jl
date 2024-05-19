
function outerenc(m=3)
    kern = (3,3)
    s = (2,2)
    θ_conv = Chain(Conv(kern,1 => 3,relu,stride=s),
                Conv(kern,3 => 6,relu,stride=s),
                Conv(kern,6 => 9,relu,stride=s),
                Conv((2,2),9 => 12,relu))

    θ_mlp = Chain(Dense(12 => 6,relu),
                Dense(6 => m,relu))

    θ = Chain(θ_conv,
                    x->reshape(x,12,:),
                    θ_mlp)
    return θ
end

function outerclassifier(m=3)
    π = Chain(Dense(m => 5,relu),
                    Dense(5 => 10,relu),
                    softmax)
    return π
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
    return ϕ
end
