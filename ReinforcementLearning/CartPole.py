import gym

def get_policy(obs):
    angle = obs[2]
    if angle < 0:
       return 0
    else:
       return 1

env = gym.make('CartPole-v0')

for episode in range(100):
    obs = env.reset()
    for step in range(1000):
        action = get_policy(obs)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
           break
		   
