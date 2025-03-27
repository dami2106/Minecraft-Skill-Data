from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback
from minestudio.models import load_vpt_policy, VPTPolicy
import helpers as hp
import numpy as np 
import os
from tqdm import tqdm  # Import tqdm for the progress bar

# Load and prepare the policy
policy = VPTPolicy.from_pretrained("CraftJarvis/MineStudio_VPT.rl_from_early_game_2x").to("cuda")
policy.eval()

# Setup the simulation environment
env = MinecraftSim(
    obs_size=(128, 128), 
    callbacks=[RecordCallback(record_path="./output", fps=30, frame_type="pov")],
    seed=np.random.randint(999999)
)
memory = None

# Set up the save folders
save_folder = 'Data/make_wooden_pickaxe'
ground_truth_folder = os.path.join(save_folder, "groundTruth")
pixel_obs_folder = os.path.join(save_folder, "pixel_obs")
os.makedirs(save_folder, exist_ok=True)
os.makedirs(ground_truth_folder, exist_ok=True)
os.makedirs(pixel_obs_folder, exist_ok=True)

# Count the number of already generated traces from the groundTruth folder.
# Here we assume that each trace produces one file starting with "minecraft_"
existing_traces = len([f for f in os.listdir(ground_truth_folder) if f.startswith("minecraft_")])
trace_nb = existing_traces

# Create a progress bar for the remaining episodes (out of 1000)
remaining_traces = 1000 - trace_nb
pbar = tqdm(total=remaining_traces, desc="Successful Episodes")

while trace_nb < 1000:
    try: 
        info_list = []
        obs_list = []
        obs, info = env.reset()
        
        info_list.append(info['inventory'])
        obs_list.append(obs.copy())
        done = False

        for i in range(800):
            action, memory = policy.get_action(obs, memory, input_shape='*')
            obs, reward, terminated, truncated, info = env.step(action)
            info_list.append(info['inventory'])
            obs_list.append(obs.copy())

            if hp.check_episode_done(info['inventory'], "wooden_pickaxe", 1):
                print("Got pickaxe, stopping episode.")
                done = True
                break

        if done:
            # Convert the inventory list to skills (or any representation you need)
            skills = hp.minestudio_inv_to_skills(info_list)
            obs_array = np.array(obs_list)
            # Save the pixel observations as a numpy array file.
            np.save(os.path.join(pixel_obs_folder, f"minecraft_{trace_nb}.npy"), obs_array)
            # Save the ground truth skills to a text file.
            with open(os.path.join(ground_truth_folder, f"minecraft_{trace_nb}"), "w") as f:
                f.write(skills.rstrip("\n"))
                
            trace_nb += 1
            pbar.update(1)
            
    except Exception as e:
        # Optionally, log the exception e for debugging
        continue

pbar.close()
env.close()