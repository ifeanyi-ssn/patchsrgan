import os
import numpy as np
import torch
from skimage import io

class ImageLoader:
    def __init__(self, train_dir='train', lr_folder_name='lr_grayscale', hr_folder_name='hr_grayscale',
                 n=None, uniform=True, seed=42):
        """
        Args:
            train_dir: Root directory containing both LR and HR folders.
            lr_folder_name: Folder name for low-resolution images.
            hr_folder_name: Folder name for high-resolution images.
            n: Number of image pairs to load.
            uniform: If True → evenly spaced sampling; if False → random sampling.
            seed: Random seed (used if uniform=False).
        """
        self.train_dir = train_dir
        self.lr_folder = os.path.join(train_dir, lr_folder_name)
        self.hr_folder = os.path.join(train_dir, hr_folder_name)
        self.n = n
        self.uniform = uniform
        self.seed = seed

        self.lr_images = None
        self.hr_images = None
        self.lr_input = None
        self.hr_input = None
        self.lr_shape = None
        self.hr_shape = None

    def _select_indices(self, total):
        """Return uniformly spaced or random indices without loading images."""
        if self.n is None or self.n >= total:
            return list(range(total))
        if self.uniform:
            # Uniformly spaced indices across dataset
            step = total / self.n
            indices = [int(i * step) for i in range(self.n)]
        else:
            np.random.seed(self.seed)
            indices = np.random.choice(total, self.n, replace=False)
        return indices

    def load_images(self):
        """Load only n uniformly/randomly sampled image pairs into memory."""
        all_lr_files = sorted(os.listdir(self.lr_folder))
        all_hr_files = sorted(os.listdir(self.hr_folder))
        assert len(all_lr_files) == len(all_hr_files), "LR and HR folder mismatch!"

        total = len(all_lr_files)
        indices = self._select_indices(total)
        selected_lr_files = [all_lr_files[i] for i in indices]
        selected_hr_files = [all_hr_files[i] for i in indices]

        print(f"Selected {len(indices)} of {total} total images (uniform_sampling={self.uniform}).")

        lr_images, hr_images = [], []
        for lr_name, hr_name in zip(selected_lr_files, selected_hr_files):
            lr_img = io.imread(os.path.join(self.lr_folder, lr_name))
            hr_img = io.imread(os.path.join(self.hr_folder, hr_name))
            lr_images.append(lr_img)
            hr_images.append(hr_img)

        self.lr_images = np.array(lr_images, dtype=np.float32)
        self.hr_images = np.array(hr_images, dtype=np.float32)
        return self.lr_images, self.hr_images

    def normalize_images(self):
        """Normalize images to [0,1]."""
        if self.lr_images is None or self.hr_images is None:
            raise ValueError("Call load_images() first.")
        self.lr_images /= 255.0
        self.hr_images /= 255.0
        return self.lr_images, self.hr_images

    def create_input_tensors(self):
        """Convert numpy arrays to PyTorch tensors with channel dim."""
        if self.lr_images is None or self.hr_images is None:
            raise ValueError("Call load_images() first.")

        # Handle 2D or 3D images
        if self.hr_images.ndim == 4 and self.lr_images.ndim == 4:  # (N,D,H,W)
            self.hr_shape = (1, *self.hr_images.shape[1:])
            self.lr_shape = (1, *self.lr_images.shape[1:])
        elif self.hr_images.ndim == 3 and self.lr_images.ndim == 3:  # (N,H,W)
            self.hr_shape = (1, *self.hr_images.shape[1:])
            self.lr_shape = (1, *self.lr_images.shape[1:])
        else:
            raise ValueError(f"Unexpected shapes: LR={self.lr_images.shape}, HR={self.hr_images.shape}")

        self.lr_input = torch.tensor(self.lr_images).unsqueeze(1)
        self.hr_input = torch.tensor(self.hr_images).unsqueeze(1)
        return self.lr_input, self.hr_input, self.lr_shape, self.hr_shape

    def load_image_as_tensors(self):
        """Full pipeline: selective load → normalize → tensor."""
        self.load_images()
        self.normalize_images()
        return self.create_input_tensors()
