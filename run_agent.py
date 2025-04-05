from argparse import ArgumentParser
import pickle
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS
import gym
import helpers as hp
import os
from collections import Counter
import json
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

    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--target-item", type=str, default='wooden_pickaxe')
    parser.add_argument("--max-steps", type=int, default=650)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="Data/Test")

    args = parser.parse_args()
    agent, env = setup_agent_env(args)

    os.makedirs(args.save_dir, exist_ok=True)
    obs_dir = os.path.join(args.save_dir, "observations")
    groundTruth_dir = os.path.join(args.save_dir, "groundTruth")
    mapping_dir = os.path.join(args.save_dir, "mapping")
    os.makedirs(mapping_dir, exist_ok=True)
    os.makedirs(groundTruth_dir, exist_ok=True)
    os.makedirs(obs_dir, exist_ok=True)

    # ✅ Added for resumability: check existing groundTruth files
    existing_files = [
        f for f in os.listdir(groundTruth_dir) if f.startswith("minecraft_")
    ]
    existing_ids = {
        int(f.replace("minecraft_", "")) for f in existing_files if f.replace("minecraft_", "").isdigit()
    }
    curr_episode_nb = max(existing_ids, default=0)
    print(f"Resuming from episode {curr_episode_nb + 1}")

    while curr_episode_nb < args.episodes:
        next_episode = curr_episode_nb + 1
        # ✅ Skip already completed episodes
        if next_episode in existing_ids:
            print(f"Episode {next_episode} already exists, skipping.")
            curr_episode_nb += 1
            continue

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
            print(f"Episode {next_episode} finished")
            curr_episode_nb += 1

            groundTruth = hp.find_inventory_activity(inv_list)
            with open(os.path.join(groundTruth_dir, f"minecraft_{curr_episode_nb}"), "w") as f:
                f.write(groundTruth.rstrip("\n"))

            obs_list = np.array(obs_list)
            obs_path = os.path.join(obs_dir, f"minecraft_{curr_episode_nb}.npy")
            np.save(obs_path, obs_list)



    env.close()

    with open(os.path.join(args.save_dir, "trace_config.json"), "w") as f:
        json.dump(
            {
                "parameters": vars(args)
            },
            f,
            indent=4,
        )


    print("Done.")
