from argparse import ArgumentParser
import os
from collections import Counter
import json


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--data-dir", type=str, default="Data/Test")

    args = parser.parse_args()

    episode_lengths = []
    ground_truth_distribution = Counter()
    ground_truth_skills = set()

    # List all the files in the args.data_dir directory
    files = os.listdir(args.data_dir + "/groundTruth")
    mapping_dir = os.path.join(args.data_dir, "mapping")
    os.makedirs(mapping_dir, exist_ok=True)

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

    #Create a mapping of i -> skill 
    skill_mapping = {i: skill for i, skill in enumerate(ground_truth_skills)}
    # Save the mapping to a txt file in the form i skill\n
    with open(os.path.join(mapping_dir, "mapping.txt"), "w") as f:
        for i, skill in skill_mapping.items():
            line = f"{i} {skill}"
            if i < len(skill_mapping) - 1:
                f.write(line + "\n")
            else:
                f.write(line)