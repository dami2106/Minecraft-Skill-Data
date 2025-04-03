from argparse import ArgumentParser
import pickle
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS
import gym
import helpers as hp
import os 
from collections import Counter
import json
from argparse import ArgumentParser
import pickle
from PIL import Image
import numpy as np

class InventoryDoneWrapper(gym.Wrapper):
    def __init__(self, env, target_item: str):
        super().__init__(env)
        self.target_item = target_item
        self._item_collected = False

    def reset(self, **kwargs):
        self._item_collected = False
        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        inventory = obs.get("inventory", {})
        item_count = inventory.get(self.target_item, 0)
        if item_count > 0:
            done = True
            self._item_collected = True

        return obs, reward, done, info




def setup_agent_env(args):
    model = args.model
    weights = args.weights
    target_item = args.target_item

    env = HumanSurvival(**ENV_KWARGS).make()
    env = InventoryDoneWrapper(env, target_item=target_item)

    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    return agent, env 

if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--target-item", type=str, default='wooden_pickaxe', help="Inventory item that terminates the episode when acquired.")
    parser.add_argument("--max-steps", type=int, default=650, help="Max nb of steps to take")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to save")
    parser.add_argument("--save-dir", type=str, default="Data/Test", help="Save directory ")

    args = parser.parse_args()
    agent, env = setup_agent_env(args)

    os.makedirs(args.save_dir, exist_ok=True)
    obs_dir = os.path.join(args.save_dir, "observations")
    groundTruth_dir = os.path.join(args.save_dir, "groundTruth")
    mapping = os.path.join(args.save_dir, "mapping")
    os.makedirs(mapping, exist_ok=True)
    os.makedirs(groundTruth_dir, exist_ok=True)
    os.makedirs(obs_dir, exist_ok=True)

    curr_episode_nb = 0
    episode_lengths = []
    ground_truth_distribution = Counter()
    ground_truth_skills = set()

    while curr_episode_nb < args.episodes:
        obs = env.reset()
        inv_list = [obs['inventory'].copy()]
        obs_list = [hp.resize_and_format(obs['pov'])]

        for _ in range(args.max_steps):
            minerl_action = agent.get_action(obs)
            obs, reward, done, info = env.step(minerl_action)

            inv_list.append(obs['inventory'].copy())
            obs_list.append(hp.resize_and_format(obs['pov']))

            if done:
                break

        if done:
            print(f"Episode {curr_episode_nb} finished")
            curr_episode_nb += 1

            groundTruth = hp.find_inventory_activity(inv_list)
            with open(args.save_dir + f"/groundTruth/minecraft_{curr_episode_nb}", "w") as f:
                f.write(groundTruth.rstrip("\n"))

            episode_length = len(obs_list)
            episode_lengths.append(episode_length)

            obs_list = np.array(obs_list)        
            obs_path = os.path.join(obs_dir, f"minecraft_{curr_episode_nb}.npy")
            np.save(obs_path, obs_list)

            for skill in groundTruth.split("\n"):
                if skill:
                    ground_truth_distribution[skill] += 1
                    ground_truth_skills.add(skill)

            
            
            

    env.close()

    stats = {
        "min_episode_length": min(episode_lengths) if episode_lengths else 0,
        "avg_episode_length": sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0,
        "max_episode_length": max(episode_lengths) if episode_lengths else 0,
        "ground_truth_distribution": dict(ground_truth_distribution)
    }

    with open(os.path.join(args.save_dir, "trace_config.json"), "w") as f:
        json.dump(
            {
                "parameters": vars(args),
                "stats": stats,
            },
            f,
            indent=4,
        )
    
    #In the mapping file, save a mapping of the ground truth with a number to the left : 0 skill_1
    with open(args.save_dir + "/mapping/mapping.txt", "w") as f:
        for i, skill in enumerate(ground_truth_skills):
            f.write(f"{i} {skill}\n")

    print("Done.")
