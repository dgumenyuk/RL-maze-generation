from itertools import combinations
import json
from stable_baselines3 import PPO

from common.calculate_novelty import calculate_novelty
from environment import RobotEnv

policy = "MlpPolicy"
#policy = "CnnPolicy"

final_results = {}
final_novelty = {}
scenario_list = []
novelty_list = []
if __name__ == "__main__":
    environ = RobotEnv(policy)

    model_save_path = "2023-01-27_models\\2023-01-27-rl_model-0_240000_steps.zip"
     #".\\2023-01-21-rl_model-0_220000_steps.zip"#"rl_model_09-24_mlp_v0_0_500000_steps.zip"
    model = PPO.load(model_save_path)

    episodes = 60

    environ.evaluate = True

    i = 0
    results = []
    m = 0
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
