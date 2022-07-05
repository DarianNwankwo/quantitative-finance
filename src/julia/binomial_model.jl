using CSV
using DataFrames
using Statistics


function binomial_model(S::F, u::F, v::F, p::F, t::Int) where F<:Float64
    leaf_nodes = []
    probs = []

    for h in 1:t+1
        coeff = u^h * v^(t-h)
        push!(leaf_nodes, coeff * S)

        prob = binomial(big(t), big(h)) * p^h * (1-p)^(t-h)
        push!(probs, prob)
    end

    return leaf_nodes, probs
end


function log_binomial_model(S::F, u::F, v::F, p::F, t::Int) where F<:Float64
    leaf_nodes = []
    log_probs = []

    for h in 1:t+1
        coeff = u^h * v^(t-h)
        push!(leaf_nodes, coeff * S)

        log_prob = log(binomial(big(t), big(h))) + log(p^h) + log((1-p)^(t-h))
        push!(log_probs, log_prob)
    end

    return leaf_nodes, log_probs
end


function main()
    msft_df = CSV.read("../data/msft_per_day.csv", DataFrame)
    N, d = size(msft_df)

    tomorrow = msft_df[2:N, "close"]
    today = msft_df[1:N-1, "close"]
    returns = (tomorrow - today) ./ today
    S = tomorrow[end]

    timestep = 1 / 252
    avg_return, stddev = mean(returns), std(returns)
    drift = avg_return / timestep
    volatility = stddev / √timestep
    u = 1 + stddev
    v = 1 - stddev
    p = .5 + drift * √timestep / (2volatility)
    t = 45

    leaf_nodes, probs = binomial_model(S, u, v, p, t)
    leaf_nodes, log_probs = log_binomial_model(S, u, v, p, t)
end


main()