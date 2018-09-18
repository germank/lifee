using Stats
using Base.Iterators
using Plots
using Random
include("bandits/bandits.jl")
using .Bandits

import Base.sum

start_energy = 1
N=1000
max_iter = 5000

struct Bandit
    probs::Vector{Real}
end

function pull(b::Bandit, lever::Integer)
    return rand() < b.probs[lever]
end

function pull(b::Bandit)
    return [rand() < p for p in b.probs]
end


function main()
    global N
    env = FiveDriftingArms()#TwoSeasonalArms(2)

    algorithms = Dict("Life" => Life(N, start_energy, n_arms(env)),
                      "AnnealingEpsilonGreedy" => AnnealingEpsilonGreedy(n_arms(env)),
                      "AnnealingSoftmax" => AnnealingSoftmax(n_arms(env)),
                      "UCB" => UCB(n_arms(env)))
                       
    for (name, algo) in  algorithms
        initialize(algo, arms(env,1))
    end
    #expectation_series = [expectation(algo, L)]
    R = Dict(x => [0] for x in keys(algorithms))
    for i in 1:max_iter
        for (algo_name, algo) in algorithms
            rewards = draw(arms(env,i))
            lever = select_arm(algo)
            reward = rewards[lever]
            println("i=$i, lever=$lever, reward=$reward, algo=$algo") 
            update(algo, lever, reward)
            push!(R[algo_name], R[algo_name][end] + (maximum(rewards) - reward))
        end
        #push!(expectation_series, expectation(population, L))
    end
    #expectation_series = hcat(expectation_series...)
    #println(expectation_series)
    #plot(expectation_series', label=["lever $i" for i=1:L])
    plot()
    for (algo_name, regret) in R
        plot!(regret, label=algo_name)
    end
    plot!(1:max_iter, label="upper bound")
    gui()
end

main()
