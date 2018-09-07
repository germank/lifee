using Stats
using DataStructures
using Base.Iterators
using Plots
using Random
import Base.sum

start_energy = 1
N=1000
max_pop = 1000
max_energy = 1000
max_iter = 5000
MULT=5

struct Bandit
    probs::Vector{Real}
end

function pull(b::Bandit, lever::Integer)
    return rand() < b.probs[lever]
end

function pull(b::Bandit)
    return [rand() < p for p in b.probs]
end


abstract type Player end
mutable struct StochasticSingleLeverPlayer <: Player
    p::Real
    lever::Integer
    gamble_energy::Number
    energy::Number
    threshold::Number
    StochasticSingleLeverPlayer(p, l, ge, e) = new(p, l, ge, e, e)
    StochasticSingleLeverPlayer(p, l, ge, e, t) = new(p, l, ge, e, t)
end

struct Gamble
    lever::Integer
    energy::Number
end

function play(player::StochasticSingleLeverPlayer)
    if rand() < player.p
        return Gamble(player.lever, min(player.gamble_energy, player.energy))
    else
        return nothing
    end
end

function reproduce(player::StochasticSingleLeverPlayer, energy::Number)
    return StochasticSingleLeverPlayer(player.p, player.lever, player.gamble_energy,
                                       energy, player.threshold)
end

function random_stochastic_single_lever_population(N, E, L)
    population = [StochasticSingleLeverPlayer(rand(), rand(1:L), rand()*E, rand(1:E))
                  for _ in 1:N]
    return population
end


function play(pop::Array{<:Player, 1})
    gambles = Array{Union{Gamble, Nothing}, 1}([play(player) for player in pop])
    @assert sum(gambles) < sum(pop)
    votes = DefaultDict{Integer, Number}(0)
    for (i, g) in enumerate(gambles)
        @assert g == nothing || g.energy <= pop[i].energy
        if g != nothing 
            votes[g.lever] += g.energy
            pop[i].energy -= g.energy
        end
    end
    if length(votes) > 0
        lever = argmax(votes)
    else
        lever = nothing
    end
    return lever, gambles
end

function reward(pop, gambles)
    global MULT
    total_gamble = sum(gambles)
    total_energy = sum(pop)
    gain = min(MULT * total_gamble, max_energy - total_energy)
    for i in 1:length(gambles)
        if gambles[i] != nothing
            pop[i].energy += gambles[i].energy / total_gamble * gain
        end
    end
    return reproduce(pop)
end

function reproduce(pop)
    new_players = Array{Player, 1}([p for p in pop if p.energy > 0])
    for p in pop 
        #if length(new_players) >= max_pop break end
        #max_energy = 1000
        if p.energy > 0
            while p.energy > 2*p.threshold
                p.energy -= p.threshold
                push!(new_players, reproduce(p, p.threshold))
            end
        end
    end
    #if length(new_players) > max_pop
        #new_players = shuffle(new_players)[1:max_pop]
        #new_players = sort(new_players, by = p -> p.energy, rev = true)[1:max_pop]
    #end
    return new_players
end

function count(pop::Array{<:Player, 1}, L::Integer)
    c = []
    for i in 1:L
        push!(c, sum([1 for p in pop if p.lever == i]))
    end
    return c
end

function sum(gambles::Vector{Union{Gamble, Nothing}})
    return sum([gamble.energy for gamble in gambles if gamble != nothing])
end

function sum(pop::Array{<:Player, 1})
    return sum([p.energy for p in pop])
end

function expectation(pop::Array{<:Player, 1}, L::Integer)
    c = []
    for i in 1:L
        push!(c, sum(Array{Number}([p.p*min(p.gamble_energy, p.energy) for p in pop if p.lever == i])))
    end
    return c
end

function main()
    global N
    L = 4
    bandit = Bandit(Stats.pweights([0.2, 0.2, 0.25, 0.1]))

    E = start_energy
    population = random_stochastic_single_lever_population(N, E, L)
    expectation_series = [expectation(population, L)]
    println(population)
    s = 0
    regret = [0]
    for i in 1:max_iter
        e1 = sum(population)
        ϵ = 0.001
        @assert e1 <= max_energy + ϵ
        lever, gambles = play(population)
        if lever != nothing 
            rewards = pull(bandit) 
            println("i=$i, lever=$lever, reward=$(rewards[lever]), population size=$(length(population)), gamble=$(sum(gambles)), total_energy=$(e1), expectation=$(expectation_series[end])") 
            if rewards[lever]
                population = reward(population, gambles)
                #@assert length(population) == max_pop || sum(population) ≈ e1 + sum(gambles)
                s+=1
                push!(regret, regret[end])
            elseif any(rewards)
                push!(regret, regret[end]+1)
                #@assert length(population) == max_pop || sum(population) ≈ e1 - sum(gambles)
            else
                push!(regret, regret[end])
                #@assert length(population) == max_pop || sum(population) ≈ e1 - sum(gambles)
            end
        end
        push!(expectation_series, expectation(population, L))
    end
    expectation_series = hcat(expectation_series...)
    #plot(expectation_series', label=["lever $i" for i=1:L])
    plot(regret)
    plot!(1:max_iter)
    gui()
end

main()
