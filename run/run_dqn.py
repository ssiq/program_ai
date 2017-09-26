from gym_envs import code_env
from model.dqn import DQNModel

DQNModel.init_para()
dqn = DQNModel()

env = code_env.CodeEnv()

while True:
    # initial observation

    obs = env.reset()
    dqn.reset()
    while True:
        # RL choose action based on obs
        action = dqn.choose_action(obs)
        new_obs, reward, done, info = env.step(action)
        dqn.store_transition(obs, action, reward, done, new_obs)

        dqn.train()

        obs = new_obs
        if done:
            break
