import numpy as np
import logging
from utilities import normalize_to_range

def her_evaluating(args, env, agent, episode):
    # set goals
    tar_x = normalize_to_range(np.clip(env.tarPosition[0], -6.2, 1.82), -6.2, 1.82, -1, 1)
    tar_z = normalize_to_range(np.clip(env.tarPosition[1], -4.92, 1.99), -4.92, 1.99, -1, 1)
    tar_ang = normalize_to_range(np.clip(env.tarPosition[2], -3.14, 3.14), -3.14, 3.14, -1, 1)
    pendulum_goal = np.array([tar_x, tar_z, tar_ang], dtype=np.float32)
    goals = pendulum_goal

    if (args.mode == 'test'):
        agent.load_models()
    env.reset()
    observation = env.get_observations()
    # observation = env.random_initialization(rb_node=env.getFromDef("hunter_v1"))
    env.disRewardOld = env.disReward
    ep_r = 0
    for r in range(args.target_step):
        state = np.float32(observation)
        # concat goals
        state = np.concatenate((state, goals))
        action = agent.choose_action(state)

        new_observation, reward, done, info = env.step(action)

        ep_r = ep_r + reward

        observation = new_observation

        if done:
            break

    print("One episode test's Return: {}".format(ep_r))
    logging.info("One episode test's Return: {}".format(ep_r))

    return ep_r