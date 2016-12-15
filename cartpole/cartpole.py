import gym
import numpy as np


def run_episode(env, params):
    obs = env.reset()
    reward_sum = 0
    for _ in range(200):
        action = 0 if np.matmul(params, obs) < 0 else 1
        obs, reward, done, info = env.step(action)
        reward_sum += reward
        if done:
            break
    return reward_sum

def random_search(env):
    best_params = None
    best_reward = 0
    for _ in range(10000):
        params = np.random.rand(4)*2 -1
        reward = run_episode(env, params)
        if reward > best_reward:
            best_reward = reward
            best_params = params
            if reward >= 200:
                break
    return best_params, best_reward

def hill_climbing(env):
    best_reward = 0
    # random initialization
    params = np.random.rand(4)*2 -1
    # noise scaling factor
    noise_scaling = 0.1
    for _ in range(10000):
        noise = np.random.rand(4)*2 -1
        params_new = params + (noise*noise_scaling)
        reward = run_episode(env, params)
        if reward > best_reward:
            best_reward = reward
            params = params_new
            if reward >= 200:
                break
    return params, best_reward

def render(env, params):
    for _ in range(1000):
        env.render()
        obs = env.reset()
        action = 0 if np.matmul(params, obs) < 0 else 1
        env.step(action)

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    #best_params, best_reward = random_search(env)
    best_params, best_reward = hill_climbing(env)
    print('Best params', best_params)
    print('Best reward', best_reward)
    render(env,best_params)
