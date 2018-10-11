include("bandits.jl")
using .Bandits

arm1 = BernoulliArm(0.7)
draw(arm1)

arm2 = BernoulliArm(0.2)
draw(arm2)

arms2 = [arm1, arm2]

algo = EpsilonGreedy(0.7, 2)

initialize(algo, arms2)

horizon = 100

s = ""
for t = 1:horizon
  global s
  chosen_arm = select_arm(algo)
  reward = draw(arms2[chosen_arm])
  s *= string(reward)
  update(algo, chosen_arm, reward)
end
println(s)

initialize(algo, arms2)
