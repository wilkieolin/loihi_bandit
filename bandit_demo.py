# Q-Tracker based e-Greedy Demo for Loihi
# Wilkie Olin-Ammentorp, Yury Sokolov, Maxim Bazhenov

import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import bandit as b
import matplotlib.pyplot as plt

#constants
numArms = 10
excess = 0.2
neuronsPerArm = 1
epsilon = 0.08
tEpoch = 128
epochs = 2000

#setup and run bandit
p_reward = b.pick_weights(numArms, excess)
bestarm = np.argmax(p_reward)
btest = b.bandit(numArms, neuronsPerArm, tEpoch, epochs, probabilities=p_reward, epsilon=epsilon, recordWeights=False)
(choices, rewards, spikes) = btest.run(epochs)
btest.stop()

print("Plotting results...")

#plot
fig = plt.figure()
plt.scatter(np.arange(epochs), choices, alpha=0.1)
plt.ylabel("Arm Chosen")
plt.xlabel("Time step")
fig.savefig("choices_over_time.png")

fig = plt.figure()
plt.plot(p_reward/100)
plt.hist(choices, bins=np.arange(numArms)-0.5, density=True)
plt.xlabel("Arm")
plt.ylabel("Proportion chosen & Probability of Reward")
fig.savefig("choices_reward_histogram.png")

fig = plt.figure()
plt.plot(np.convolve([x == bestarm for x in choices], np.ones(5), mode='valid')/5)
plt.xlabel("Time step")
plt.ylabel("Mean Optimal Action over last 5 epochs")
fig.savefig("moa.png")
