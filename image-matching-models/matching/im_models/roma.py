import torch
import torchvision.transforms as tfm
from kornia.augmentation import PadTo
from kornia.utils import tensor_to_image
import tempfile
import os
from pathlib import Path


from matching import BaseMatcher, THIRD_PARTY_DIR
from matching.utils import add_to_path

add_to_path(THIRD_PARTY_DIR.joinpath("RoMa"))
from romatch import roma_outdoor, tiny_roma_v1_outdoor

from PIL import Image
from skimage.util import img_as_ubyte


def _get_project_temp_dir():
    """
    Get or create a project-local temporary directory.
    This ensures temp files are created in the project directory, not C drive.
    """
    # Check if environment variable is already set (from setup_temp_dir)
    if "TEMP" in os.environ and os.environ["TEMP"]:
        temp_path = Path(os.environ["TEMP"])
        if temp_path.exists() or temp_path.parent.exists():
            return temp_path
    
    # Fallback: Try to find project root by looking for common project markers
    current_file = Path(__file__).resolve()
    project_root = current_file
    # Navigate up to find project root (look for image-matching-models or Visual-Place-Recognition-Project)
    for _ in range(10):  # Limit search depth
        if project_root.name in ["image-matching-models", "Visual-Place-Recognition-Project"]:
            break
        if project_root.parent == project_root:  # Reached filesystem root
            break
        project_root = project_root.parent
    
    # Create temp directory in project root
    temp_dir = project_root / ".temp"
    temp_dir.mkdir(exist_ok=True)
    return temp_dir


class RomaMatcher(BaseMatcher):
    dino_patch_size = 14
    coarse_ratio = 560 / 864

    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.roma_model = roma_outdoor(device=device)
        self.max_keypoints = max_num_keypoints
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.roma_model.train(False)

    def compute_padding(self, img0, img1):
        _, h0, w0 = img0.shape
        _, h1, w1 = img1.shape
        pad_dim = max(h0, w0, h1, w1)

        self.pad = PadTo((pad_dim, pad_dim), keepdim=True)

    def preprocess(self, img: torch.Tensor, pad=False) -> Image:
        if isinstance(img, torch.Tensor) and img.dtype == (torch.float):
            img = torch.clamp(img, -1, 1)
        if pad:
            img = self.pad(img)
        img = tensor_to_image(img)
        pil_img = Image.fromarray(img_as_ubyte(img), mode="RGB")
        # Use project temp directory instead of system temp
        temp_dir = _get_project_temp_dir()
        temp = tempfile.NamedTemporaryFile("w+b", suffix=".png", delete=False, dir=str(temp_dir))
        pil_img.save(temp.name, format="png")
        return temp, pil_img.size

    def _forward(self, img0, img1, pad=False):
        if pad:
            self.compute_padding(img0, img1)
        img0_temp, img0_size = self.preprocess(img0)
        img1_temp, img1_size = self.preprocess(img1)
        w0, h0 = img0_size
        w1, h1 = img1_size

        try:
            warp, certainty = self.roma_model.match(img0_temp.name, img1_temp.name, batched=False, device=self.device)

            matches, certainty = self.roma_model.sample(warp, certainty, num=self.max_keypoints)
            mkpts0, mkpts1 = self.roma_model.to_pixel_coordinates(matches, h0, w0, h1, w1)

            return mkpts0, mkpts1, None, None, None, None
        finally:
            # Always clean up temporary files, even if an exception occurs
            try:
                img0_temp.close()
            except:
                pass
            try:
                img1_temp.close()
            except:
                pass
            try:
                Path(img0_temp.name).unlink(missing_ok=True)
            except:
                pass
            try:
                Path(img1_temp.name).unlink(missing_ok=True)
            except:
                pass


class TinyRomaMatcher(BaseMatcher):

    def __init__(self, device="cpu", max_num_keypoints=2048, *args, **kwargs):
        super().__init__(device, **kwargs)
        self.roma_model = tiny_roma_v1_outdoor(device=device)
        self.max_keypoints = max_num_keypoints
        self.normalize = tfm.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.roma_model.train(False)

    def preprocess(self, img):
        return self.normalize(img).unsqueeze(0)

    def _forward(self, img0, img1):
        img0 = self.preprocess(img0)
        img1 = self.preprocess(img1)

        h0, w0 = img0.shape[-2:]
        h1, w1 = img1.shape[-2:]

        # batch = {"im_A": img0.to(self.device), "im_B": img1.to(self.device)}
        warp, certainty = self.roma_model.match(img0, img1, batched=False)

        matches, certainty = self.roma_model.sample(warp, certainty, num=self.max_keypoints)
        mkpts0, mkpts1 = self.roma_model.to_pixel_coordinates(matches, h0, w0, h1, w1)

        return mkpts0, mkpts1, None, None, None, None
