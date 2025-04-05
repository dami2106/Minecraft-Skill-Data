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

if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--data-dir", type=str, default="Data/Test")

    args = parser.parse_args()

    episode_lengths = []
    ground_truth_distribution = Counter()
    ground_truth_skills = set()

    # List all the files in the args.data_dir directory
    files = os.listdir(args.data_dir + "/groundTruth")

    for file in files:
        # Open the JSON file and load its contents
        with open(os.path.join(args.data_dir + "/groundTruth", file), 'r') as f:
            data = f.read()
            for skill in data.split("\n"):
                if skill:
                    ground_truth_distribution[skill] += 1
                    ground_truth_skills.add(skill)

            episode_length = len(data.split("\n"))
            episode_lengths.append(episode_length)


    stats = {
        "total_episodes": len(episode_lengths),
        "unique_skills": len(ground_truth_skills),
        "min_episode_length": min(episode_lengths) if episode_lengths else 0,
        "avg_episode_length": sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0,
        "max_episode_length": max(episode_lengths) if episode_lengths else 0,
        "ground_truth_distribution": dict(ground_truth_distribution)
    }

    # Save the stats to a JSON file
    with open(os.path.join(args.data_dir, "stats.json"), 'w') as f:
        json.dump(stats, f, indent=4)