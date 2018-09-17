MULT=5
import Base.show
import Base.sum
using DataStructures

struct Gamble
    lever::Integer
    energy::Number
end

abstract type Player end
mutable struct Life <: BanditAlgorithm
    population::Array{Player, 1}
    last_gambles::Array{Union{Gamble, Nothing}}
    n_arms::Integer
    max_energy::Number
    Life(population, n_arms) = new(population, Array{Union{Gamble,Nothing},1}(), n_arms, 1000)
end


function initialize(algo::Life, arms)
end

function Life(N, E, n_arms)
    population = [StochasticSingleLeverPlayer(rand(), rand(1:n_arms), rand()*E, rand(1:E))
                  for _ in 1:N]
    return Life(population, n_arms)
end

function Base.show(io::IO, algo::Life)
    print(io, "Life(population_size = $(length(algo.population)), energy = $(sum(algo.population)), gambles = $(sum(algo.last_gambles)))")
end

mutable struct StochasticSingleLeverPlayer <: Player
    p::Real
    lever::Integer
    gamble_energy::Number
    energy::Number
    threshold::Number
    StochasticSingleLeverPlayer(p, l, ge, e) = new(p, l, ge, e, e)
    StochasticSingleLeverPlayer(p, l, ge, e, t) = new(p, l, ge, e, t)
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


function sum(gambles::Vector{Union{Gamble, Nothing}})
    return sum([gamble.energy for gamble in gambles if gamble != nothing])
end

function sum(pop::Array{<:Player, 1})
    return sum([p.energy for p in pop])
end

function count(pop::Array{<:Player, 1}, L::Integer)
    c = []
    for i in 1:L
        push!(c, sum([1 for p in pop if p.lever == i]))
    end
    return c
end

function expectation(pop::Array{<:Player, 1}, L::Integer)
    c = []
    for i in 1:L
        push!(c, sum(Array{Number}([p.p*min(p.gamble_energy, p.energy) for p in pop if p.lever == i])))
    end
    return c
end


function select_arm(algo::Life)
    gambles = Array{Union{Gamble, Nothing}, 1}([play(player) for player in algo.population])
    @assert sum(gambles) < sum(algo.population)
    e1 = sum(algo.population)
    ϵ = 0.001
    @assert e1 <= algo.max_energy + ϵ
    votes = DefaultDict{Integer, Number}(0)
    for (i, g) in enumerate(gambles)
        @assert g == nothing || g.energy <= algo.population[i].energy
        if g != nothing 
            votes[g.lever] += g.energy
            algo.population[i].energy -= g.energy
        end
    end
    if length(votes) > 0
        lever = argmax(votes)
    else
        lever = randint(1:algo.n_arms)
    end
    algo.last_gambles = gambles
    return lever
end

function update(algo::Life, chosen_arm::Int64, reward::Real)
    global MULT
    gambles = algo.last_gambles
    total_gamble = sum(gambles)
    total_energy = sum(algo.population)
    gain = min(MULT * reward * total_gamble, algo.max_energy - total_energy)
    for i in 1:length(gambles)
        if gambles[i] != nothing
            algo.population[i].energy += gambles[i].energy / total_gamble * gain
        end
    end
    reproduce(algo)
end

function reproduce(algo::Life)
    new_players = Array{Player, 1}([p for p in algo.population if p.energy > 0])
    for p in algo.population
        #if length(new_players) >= max_pop break end
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
    algo.population = new_players
end
