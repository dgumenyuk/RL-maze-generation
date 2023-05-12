# import gymnasium as gym
import json
import time
from itertools import combinations


from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback


import config as cf
from environment import RobotEnv
from common.calculate_novelty import calculate_novelty

if __name__ == "__main__":
    print("Starting the RL agent training")

    final_results = {}
    final_novelty = {}
    scenario_list = []
    novelty_list = []

    m = 0
    policy = "MlpPolicy"
    #policy = "CnnPolicy"
    for m in range(5):


        environ = RobotEnv(policy)

        # check_env(environ)

        checkpoint_callback_ppo = CheckpointCallback(
            save_freq=20000,
            save_path=cf.files["model_path"],
            name_prefix="2023-01-28-rl_model-" + str(m),
        )
        log_path = cf.files["logs_path"]

        start = time.time()

        model = PPO(
            policy, environ, verbose=True, ent_coef=0.005, tensorboard_log=log_path
        )  

        # Start training the agent
        model.learn(
            total_timesteps=600000,
            tb_log_name="2023-01-28-rl_model-" + str(m),
            callback=checkpoint_callback_ppo,
        ) 
        print("Training time: {}".format(time.time() - start))


        episodes = 60

        i = 0
        results = []
        while environ.episode < episodes:
            obs = environ.reset()
            done = False

            while not done:
                action, _ = model.predict(obs)
                obs, rewards, done, info = environ.step(action)
            i += 1
            # max_fitness = max(environ.all_fitness)
            max_fitness = environ.fitness


            if (max_fitness > 70) or i > 15: # 15 attepts to produce a good scenario
                print(i)
                print("Round: {}".format(environ.episode))
                print("Max fitness: {}".format(max_fitness))
                scenario = environ.state
                environ.render(scenario)
                scenario_list.append(scenario)
                environ.episode += 1
                results.append(max_fitness)
                i = 0

        final_results[str(m)] = results

        novelty_list = []
        for i in combinations(range(0, 30), 2):
            current1 = scenario_list[i[0]]
            current2 = scenario_list[i[1]]
            nov = calculate_novelty(current1, current2)
            novelty_list.append(nov)
        novelty = abs(sum(novelty_list) / len(novelty_list))

        final_novelty[str(m)] = novelty

        scenario_list = []

        with open("2023-01-21-results-ppo.txt", "w") as f:
            json.dump(final_results, f, indent=4)

        with open("2023-01-21-novelty-ppo.txt", "w") as f:
            json.dump(final_novelty, f, indent=4)
