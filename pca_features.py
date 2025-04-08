import numpy as np
import glob
from collections import Counter
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import IncrementalPCA
import joblib
import random
from argparse import ArgumentParser



import glob
import numpy as np
from sklearn.decomposition import IncrementalPCA
import psutil
import os

from tqdm import tqdm


import os
import glob
import psutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import IncrementalPCA
import joblib
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


def run_pca(args):
    mem_info = psutil.virtual_memory()
    total_mem_mb = mem_info.total / (1024 ** 2)
    print(f"Total memory available: {total_mem_mb:.2f} MB")

    numpy_files = sorted(glob.glob(args.data_dir + "/observations/*"))
    process = psutil.Process(os.getpid())

    all_shapes = [np.load(f, allow_pickle=True).shape[0] for f in numpy_files]
    total_samples = sum(all_shapes)
    example = np.load(numpy_files[0], allow_pickle=True)
    flat_shape = np.prod(example.shape[1:])
    final_array = np.empty((total_samples, flat_shape), dtype=np.float32)

    offset = 0
    img_arr = None
    for file_ in numpy_files:
        data = np.load(file_, allow_pickle=True).astype(np.uint8)
        print("data shape min and max ", data.shape, data.min(), data.max())

        data = data.astype(np.float32) / 255.0
        img_arr = data

        data = data.reshape(data.shape[0], -1)
        print("data shape min and max after proc", data.shape, data.min(), data.max())
        batch_size = data.shape[0]
        final_array[offset:offset + batch_size] = data
        print("final_array shape min and max ", final_array.shape, final_array.min(), final_array.max())
        offset += batch_size
        break

        mem = process.memory_info().rss / (1024 ** 2)
        print(f"Memory usage: {mem:.2f} MB")

    #Save img arr to gif
    img_arr = np.array(img_arr)
    # Convert the array to a list of PIL Images
    frames = [Image.fromarray((frame * 255).astype(np.uint8)) for frame in img_arr]

    # Save as a GIF
    gif_path = os.path.join(args.data_dir, "animation.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

    print(f"GIF saved to: {gif_path}")
    
    # print("Final shape:", final_array.shape)
    # mem = process.memory_info().rss / (1024 ** 2)
    # print(f"Memory usage all loaded: {mem:.2f} MB")

    # # Shuffle the data in place
    np.random.shuffle(final_array)

    # print("\nFitting PCA...")
    # pca = IncrementalPCA(n_components=100, batch_size=500)
    # pca.fit(final_array)
    # print("PCA fitted.")

    # print("Explained variance by top components:", pca.explained_variance_ratio_[:10])
    # print("Total explained variance:", pca.explained_variance_ratio_.sum())

    # # Save PCA model
    # model_path = os.path.join(args.data_dir, "pca_model.joblib")
    # joblib.dump(pca, model_path)
    # print(f"PCA model saved to: {model_path}")

    # Plotting original and reconstructed images
    sample_indices = np.random.choice(final_array.shape[0], 10, replace=False)
    sampled = final_array[sample_indices]

    print(sampled.shape)

    test_img = sampled[0].reshape(140, 140, 3)  

    print("Sampled image shape:", test_img.shape)
    print(test_img.min(), test_img.max())

    plt.imshow(test_img)
    plt.axis("off")
    plt.title("Sampled Image")
    plt.show()

    #Load pca mdoel
    # model_path = os.path.join(args.data_dir, "pca_model.joblib")
    # pca = joblib.load(model_path)

    # # Reconstruct images
    # pca_features = pca.transform(sampled)
    # reconstructed = pca.inverse_transform(pca_features)

    # # Save original vs reconstructed to PDF
    # pdf_path = os.path.join(args.data_dir, "pca_reconstruction.pdf")
    # with PdfPages(pdf_path) as pdf:
    #     for i in range(10):
    #         fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    #         for ax, img_data, title in zip(
    #             axes,
    #             [sampled[i], reconstructed[i]],
    #             ["Original", "Reconstructed"]
    #         ):
    #             img = img_data.reshape(140, 140, 3)
    #             img = np.clip(img, 0, 1)  # Ensure valid pixel range
    #             ax.imshow(img)
    #             ax.axis("off")
    #             ax.set_title(title)
    #         pdf.savefig(fig)
    #         plt.close(fig)

    # print(f"Saved PCA reconstruction comparison to: {pdf_path}")


    

# def run_pca(args):
#     numpy_files = sorted(glob.glob(args.data_dir + "/observations/*"))

#     if not numpy_files:
#         raise ValueError(f"No files found in {args.data_dir}/observations/")

#     pca = IncrementalPCA(n_components=args.components)

#     process = psutil.Process(os.getpid())

#     for idx in range(0, len(numpy_files), args.episode_batch):
#         print(f"\nProcessing files {idx} to {idx + args.episode_batch} out of {len(numpy_files)}")

#         group_files = numpy_files[idx:idx + args.episode_batch]
#         group_imgs = []

#         for file_ in group_files:
#             data = np.load(file_, allow_pickle=True).astype(np.uint8)

#             data = data.astype(np.float32) / 255.0
#             data = data.reshape(data.shape[0], -1)
#             group_imgs.append(data)

#         group_imgs = np.concatenate(group_imgs, axis=0)

#         # Log RAM usage
#         mem = process.memory_info().rss / (1024 ** 2)  # MB
#         print(f"Memory usage before partial_fit: {mem:.2f} MB")

#         pca.partial_fit(group_imgs)

#         #Extract features for this episode 

#         mem = process.memory_info().rss / (1024 ** 2)  # MB
#         print(f"Memory usage after partial_fit: {mem:.2f} MB")

#         print("\nExplained variance by top components:", pca.explained_variance_ratio_[:10])
#         print("Total explained variance:", pca.explained_variance_ratio_.sum())


#     print("\nFinalizing PCA...")
#     print("Final Explained variance by top components:", pca.explained_variance_ratio_[:10])
#     print("Final Total explained variance:", pca.explained_variance_ratio_.sum())


#     print("\nSaving PCA features...")
#     for file_ in tqdm(numpy_files):
#         data = np.load(file_, allow_pickle=True)
#         data = data.astype(np.float32) / 255.0
#         flattened_images = []
#         file_name = file_.split("/")[-1]

#         for frame in data:
#             img = frame['image']
#             img = img.astype(np.float32) / 255.0
#             img = img.flatten()

#             #Transform the image using the PCA model
#             img = pca.transform([img])[0]
#             flattened_images.append(img)

#         flattened_images = np.array(flattened_images)
#         np.save(f"{args.data_dir}/pca_features/{file_name}", flattened_images)

#     print("PCA features saved to disk.")


#     random_files = random.sample(numpy_files, 5)

#     og_imgs = []
#     reconstructed_imgs = []

#     for file_ in random_files:
#         data = np.load(file_, allow_pickle=True).astype(np.uint8)
#         #Choose a random index from 1 to the length of the data
#         random_index = random.randint(1, len(data))

#         img = data[random_index]
#         img = img.astype(np.float32) / 255.0
#         pca_image = pca.transform([img.flatten()])[0]

#         reconstructed_img = pca.inverse_transform(pca_image)
#         reconstructed_img = reconstructed_img.reshape(140, 140, 3)

#         og_imgs.append(img)
#         reconstructed_imgs.append(reconstructed_img)


#     #Plot the original images and reconstructed images side by side and save as pdf
#     fig, ax = plt.subplots(5, 2, figsize=(10, 20))

#     for i in range(5):
#         ax[i, 0].imshow(og_imgs[i])
#         ax[i, 0].set_title("Original Image")
#         ax[i, 0].axis('off')

#         ax[i, 1].imshow(reconstructed_imgs[i])
#         ax[i, 1].set_title("Reconstructed Image")
#         ax[i, 1].axis('off')

#     plt.tight_layout()
#     plt.savefig(f"{args.data_dir}/reconstructed_images.pdf")
#     plt.show()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--data-dir", type=str, default="Data/wooden_pickaxe")
    parser.add_argument("--components", type=int, default=1000)
    parser.add_argument("--episode-batch", type=int, default=20)

    args = parser.parse_args()

    # os.makedirs(args.data_dir + "/pca_features", exist_ok=True)

    run_pca(args)