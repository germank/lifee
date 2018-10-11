import Base.show
import Base.sum
using DataStructures

abstract type Agent end
abstract type Environment end

struct Bid
    value::Integer
    energy::Number
    owner::Agent
end

mutable struct Variable
    dim::Integer
    value::Integer
    energy::Number
    bids::Vector{Bid}

    Variable(dim) = new(dim, 1, 0, Vector{Bid}())
end

function place_bid!(x::Variable, value::Integer, energy::Number, owner::Agent)
    @assert energy >= 0
    push!(x.bids, Bid(value, energy, owner))
end

function update!(x::Variable)
    votes = DefaultDict{Integer, Number}(0)
    for bid in x.bids
        votes[bid.value] += bid.energy
    end
    if length(votes) > 0
        x.value = argmax(votes)
    end
end

function feedback!(x::Variable)
    # only reward bids for the correct value? (uncomment to enable)
    # If this is enforced, alternative options die out. (Could be relevant later!)
    #x.bids = [bid for bid in x.bids if bid.value == x.value]
    # diffuse the energy of the variable proportionally to the bids
    @assert x.energy >= 0
    if length(x.bids) == 0 return end
    total_bid = sum(bid.energy for bid in x.bids)
    for bid in x.bids
        @assert bid.owner.bid_variable == x
        bid.owner.energy += bid.energy / total_bid * x.energy
    end
    x.energy = 0
    empty!(x.bids)
end

BooleanVariable() = Variable(2)

AgentPopulation = Vector{Agent}

mutable struct Life <: BanditAlgorithm
    population::AgentPopulation
    environment::Environment
    max_energy::Number
    gain::Number
    n_arms::Integer
end

function Life(N, n_arms, hidden_size, gain)
    max_energy = 1.1N
    env = VariableSetEnv()
    # output variable
    push!(env.variables, Variable(n_arms))
    # hidden boolean variables
    for i = 1:hidden_size
        push!(env.variables, BooleanVariable())
    end

    population = AgentPopulation()
    for i in 1:N
        bidding_var = rand(env.variables)
        bidding_val = rand(1:bidding_var.dim)
        energy = rand()
        gambling_energy = rand() * energy
        push!(population, 
              StochasticSingleLeverAgent(rand(0:1,hidden_size),  
                                         rand(), # bias
                                         bidding_val, # lever
                                         bidding_var, # variable idx
                                         gambling_energy, # gamble energy
                                         energy
                                         )
             )
    end
    return Life(population, env, max_energy, gain, n_arms)
end

function initialize(algo::Life, arms)
end

sum0(x) = reduce(+, x; init=0)

function Base.show(io::IO, algo::Life)
    total_energy = sum(agent.energy for agent in algo.population)
    variable = algo.environment.variables[1]
    total_gambles = [sum0(bid.energy for bid in variable.bids if bid.value == i) for i in 1:algo.n_arms]
    energy = [sum0(agent.energy for agent in algo.population if agent.bid_variable == var) for var in algo.environment.variables]
    print(io, "Life(population_size = $(length(algo.population)), energy = $energy, gambles = $total_gambles, state=$(state(algo.environment))")
end

mutable struct VariableSetEnv <: Environment
    variables::Vector{Variable}
    last_gambles::Array{Union{Bid, Nothing}}

    VariableSetEnv(variables) = new(variables, [])
    VariableSetEnv() = new(Vector{Variable}())
end

function state(env::VariableSetEnv)
    return [var.value - 1 for var in env.variables[2:end]]
end

function absorb!(env::VariableSetEnv, energies)
    for (var, e) in zip(env.variables[2:end], energies)
        @assert e >= 0
        var.energy += e
    end
end

function update!(env::VariableSetEnv)
    for x in env.variables
        update!(x)
    end
end

function feedback!(env::VariableSetEnv)
    for x in env.variables
        feedback!(x)
    end
end


mutable struct StochasticSingleLeverAgent <: Agent
    w::Array{Real,1}
    b::Real
    bid_value::Integer
    bid_variable::Variable
    gamble_energy::Real
    energy::Real
    threshold::Real
    StochasticSingleLeverAgent(w, b, c, i, ge, e) = new(w, b, c, i, ge, e, e)
    function StochasticSingleLeverAgent(w, b, c, i, ge, e, t)
        @assert e > 0
        @assert ge <= e
        @assert ge <= t
        return new(w, b, c, i, ge, e, t)
    end
end

τ = 0.2
sigmoid(x) = 1/(1+exp(-(x-0.5)/τ))
linear(x) = x
import LinearAlgebra: norm
function probability(agent::StochasticSingleLeverAgent, s)
    #if length(p.w) > 0
    #    net += p.w' * (s ) / length(p.w)
    #end
    #return sigmoid(net)
    σ = 0.25
    p = 1
    if length(agent.w) > 0
        p = exp(-norm(agent.w - s) / (2 * σ ^ 2))
    end
    return agent.b * p
end

function softmax(xs::Array{<:Number, 1}, temp::Number=1)
    if length(xs) == 0 return xs end
    xs = exp.(xs / temp)
    return xs / sum(xs)
end

function act!(env::Environment, agent::StochasticSingleLeverAgent)
    @assert agent.energy >= 0
    p = probability(agent, state(env))
    if agent.energy > 0 && rand() < p
        bid_energy = min(agent.gamble_energy, agent.energy)
        agent.energy -= bid_energy
        place_bid!(agent.bid_variable, agent.bid_value, bid_energy, agent)
        free_energy = (agent.w .== state(env)) * bid_energy / length(agent.w)
        #println(agent.w,  state(env), p," ", free_energy)
        absorb!(env, free_energy)
    end
end

function reproduce(agent::StochasticSingleLeverAgent)
    new_agent_energy = agent.threshold
    agent.energy -= new_agent_energy
    @assert agent.energy > 0
    return StochasticSingleLeverAgent(agent.w, agent.b, agent.bid_value, agent.bid_variable, agent.gamble_energy,
                                       new_agent_energy, agent.threshold)
end

function update!(population::AgentPopulation)
    new_agents = Array{Agent, 1}()
    dead = []
    for (i,agent) in enumerate(population)
        if agent.energy == 0 push!(dead, i) end
        while agent.energy > 2*agent.threshold
            push!(new_agents, reproduce(agent))
        end
    end
    deleteat!(population, dead)
    append!(population, new_agents)
end

function expectation(algo::Life, variable_idx, L)
    e = DefaultDict{Integer, Float64}(0)
    for p in algo.population
        p_variable_idx = indexin([p.bid_variable], algo.environment.variables)[1]
        if p_variable_idx == variable_idx
            e[p.bid_value] += probability(p, state(algo.environment))*min(p.gamble_energy, p.energy) 
        end
    end
    ret = zeros(L)
    for i in 1:L
        if haskey(e, i)
            ret[i] = e[i]
        end
    end
    return ret
end


function select_arm(algo::Life)
    for (i, agent) in enumerate(algo.population)
        g = act!(algo.environment, agent)
    end
    update!(algo.environment)

    return algo.environment.variables[1].value
end

function update(algo::Life, chosen_arm::Int64, reward::Real)
    total_energy = sum(agent.energy for agent in algo.population)
    gain = min(max(0, algo.max_energy - total_energy), algo.gain * reward)
    @assert gain >= 0
    algo.environment.variables[1].energy += gain
    feedback!(algo.environment)
    update!(algo.population)
end

