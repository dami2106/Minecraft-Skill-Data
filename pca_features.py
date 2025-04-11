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
import cv2
import matplotlib.animation as animation

def run_pca(args, every_n=3):  # NEW: add sampling frequency as a parameter
    mem_info = psutil.virtual_memory()
    total_mem_mb = mem_info.total / (1024 ** 2)
    print(f"Total memory available: {total_mem_mb:.2f} MB")

    numpy_files = sorted(glob.glob(args.data_dir + "/observations/*"))
    process = psutil.Process(os.getpid())

    # First pass: compute total number of sampled frames
    total_samples = 0
    for f in numpy_files:
        shape = np.load(f, allow_pickle=True).shape[0]
        sampled_indices = list(range(0, shape, every_n))
        if sampled_indices[-1] != shape - 1:
            sampled_indices.append(shape - 1)
        total_samples += len(sampled_indices)
    
    print(f"Total number of sampled frames: {total_samples}")

    # Example shape for flattening
    example = np.load(numpy_files[0], allow_pickle=True)
    flat_shape = np.prod(example.shape[1:])
    print(f"Flattened shape: {flat_shape}")
    final_array = np.empty((total_samples, flat_shape), dtype=np.float32)

    offset = 0
    for file_ in numpy_files:
        data = np.load(file_, allow_pickle=True).astype(np.uint8)
        print("Loaded raw shape:", data.shape)

        # Sample every nth frame and ensure final frame is included
        sampled_indices = list(range(0, data.shape[0], every_n))
        if sampled_indices[-1] != data.shape[0] - 1:
            sampled_indices.append(data.shape[0] - 1)

        data = data[sampled_indices]
        data = data.astype(np.float32) / 255.0
        print("Sampled shape after scaling:", data.shape)

        data = data.reshape(data.shape[0], -1)

        print("Data shape after flattening:", data.shape)

        # Store into final PCA array
        batch_size = data.shape[0]
        final_array[offset:offset + batch_size] = data
        offset += batch_size

        print(f"Current final_array slice: {offset}/{total_samples}")

    print("Final shape:", final_array.shape)
    mem = process.memory_info().rss / (1024 ** 2)
    print(f"Memory usage all loaded: {mem:.2f} MB")

    # Shuffle the data in place
    np.random.shuffle(final_array)

    print("\nFitting PCA...")
    pca = IncrementalPCA(n_components=1000, batch_size=20000)
    pca.fit(final_array)
    print("PCA fitted.")

    # print("Explained variance by top components:", pca.explained_variance_ratio_[:10])
    # print("Total explained variance:", pca.explained_variance_ratio_.sum())

    # # Save PCA model
    # model_path = os.path.join(args.data_dir, "pca_model.joblib")
    # joblib.dump(pca, model_path)
    # print(f"PCA model saved to: {model_path}")

    # # Plotting original and reconstructed images
    # sample_indices = np.random.choice(final_array.shape[0], 10, replace=False)
    # sampled = final_array[sample_indices]

    # print(sampled.shape)

    # test_img = sampled[0].reshape(140, 140, 3)  

    # print("Sampled image shape:", test_img.shape)
    # print(test_img.min(), test_img.max())

    # plt.imshow(test_img)
    # plt.axis("off")
    # plt.title("Sampled Image")
    # plt.show()

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
    parser.add_argument("--batch-size", type=int, default=8000)

    args = parser.parse_args()

    # os.makedirs(args.data_dir + "/pca_features", exist_ok=True)

    run_pca(args)