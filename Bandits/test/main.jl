using Base.Iterators
using Plots
using Random
using Bandits

import Base.sum

Random.seed!(1)
N = 1000
max_iter = 2000
hidden_size = 5
gain = 0.5
env = TwoSeasonalArms(1)
algorithms = Dict("Life" => Life(N, n_arms(env), hidden_size, gain),
                  "AnnealingEpsilonGreedy" => AnnealingEpsilonGreedy(n_arms(env)),
                  "AnnealingSoftmax" => AnnealingSoftmax(n_arms(env)),
                  "UCB" => UCB(n_arms(env))
                 )

struct Bandit
    probs::Vector{Real}
end

function pull(b::Bandit, lever::Integer)
    return rand() < b.probs[lever]
end

function pull(b::Bandit)
    return [rand() < p for p in b.probs]
end


function main(algorithms)
    for (name, algo) in  algorithms
        initialize(algo, arms(env,1))
    end
    expectation_series = [expectation(algorithms["Life"], 1, n_arms(env))]
    R = Dict(x => [0] for x in keys(algorithms))
    s=0
    for i in 1:max_iter
        for (algo_name, algo) in algorithms
            rewards = draw(arms(env,i))
            lever = select_arm(algo)
            reward = rewards[lever]
            s+=reward
            if algo_name == "Life"
                println("i=$i, lever=$lever, reward=$reward, algo=$algo") 
            end
            update(algo, lever, reward)
            push!(R[algo_name], R[algo_name][end] + reward)
        end
        push!(expectation_series, expectation(algorithms["Life"], 1, n_arms(env)))
    end
    pyplot()
    plot_exp = true
    plot_reward = true
    if plot_exp
        plt = plot(reuse = false)
        expectation_series = hcat(expectation_series...)
        for i in 1:size(expectation_series, 1)
            plt=plot!(expectation_series[i,100:end], label="lever $i", show=true)
        end
        display(plt)
    end
    if plot_reward
        plt2 = plot(reuse = false)
        for (algo_name, regret) in R
            plot!(regret, label=algo_name)
        end
        plot!(1:max_iter, label="upper bound")
        gui(plt2)
    end
end

main(algorithms)
