using Flux, Transformers, Transformers.HuggingFace, HuggingFaceDatasets, Distributions
using JLD2,CSV,DataFrames

l1 = 0.0001
η = 0.0001

gpt2enc, gpt2 = hgf"gpt2";

gpt2 = gpt2 |> gpu

dat = load_dataset("NeelNanda/pile-small-tokenized-2b")
pile_small = dat["train"]
jldsave("pile-small-tokenized.jld2"; pile_small)

b=50000

for i in 1:b:length(pile_small)
    df = DataFrame(pile_small[i:min(i+b-1)])
    CSV.write("pile-small-tokenized-2b.csv",df,append=true)
end
