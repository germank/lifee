struct FiveDriftingArms <: BanditEnvironment
    period::Float64
    FiveDriftingArms(period=100) = new(period)
end

function arms(env::FiveDriftingArms, trial::Int64)
  [
    BernoulliArm(((0 + trial) % env.period) / env.period),
    BernoulliArm(((10 + trial) % env.period) / env.period),
    BernoulliArm(((20 + trial) % env.period) / env.period),
    BernoulliArm((env.period - ((1 + trial) % env.period)) / env.period),
    BernoulliArm((env.period - trial % env.period) / env.period)
   ]
end

function n_arms(env::FiveDriftingArms)
  5
end
