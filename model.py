import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# resnet backbone 
_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
_resnet.eval()


#image transform bc ResNet expects images in the format it was trained on (224x224, normalized)
_resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# MiDaS depth model 
_midas = None
_midas_transform = None
_midas_load_attempted = False

# Haar cascade + crop stats — used by all four stages
_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
_crop_stats = {"detected": 0, "fallback": 0}


# load frames

def _load_frames(path: str, num_frames: int) -> list[np.ndarray]:
    """Evenly sample num_frames RGB frames from a video file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    idxs = np.linspace(0, total - 1, num=min(num_frames, total), dtype=int) if total > 0 else None

    frames = []
    if idxs is not None:
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, bgr = cap.read()
            if ok:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    else:
        while len(frames) < num_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    cap.release()
    if not frames:
        raise ValueError(f"No frames extracted from: {path}")
    return frames

#filetype checks
def _is_image(path: str) -> bool:
    return path.lower().endswith((".jpg", ".jpeg", ".png"))


def _is_video(path: str) -> bool:
    return path.lower().endswith((".mp4", ".avi", ".mov"))


def _check_path(path: str) -> None:
    if not _is_image(path) and not _is_video(path):
        raise ValueError(f"Unsupported file type: {path}")


def _load_midas() -> None:
    global _midas, _midas_transform, _midas_load_attempted
    if _midas is not None:
        return
    if _midas_load_attempted:
        raise RuntimeError(
            "MiDaS failed on a previous attempt. "
            "Make sure timm is installed: pip install timm"
        )
    _midas_load_attempted = True
    print("[depth_stage] Loading MiDaS small (first call)...")
    _midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    _midas.eval()
    t = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    _midas_transform = t.small_transform
    print("[depth_stage] MiDaS loaded.")

#crop face
#	1.	Convert to grayscale
#	2.	Detect faces
#	3.	Pick largest face
#	4.	Add padding (20%)
#	5.	Crop image
def _crop_face(rgb: np.ndarray, margin: float = 0.20) -> np.ndarray:

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) == 0:
        _crop_stats["fallback"] += 1
        return rgb

    _crop_stats["detected"] += 1
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    H, W = rgb.shape[:2]
    pad_x, pad_y = int(w * margin), int(h * margin)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(W, x + w + pad_x), min(H, y + h + pad_y)

    return rgb[y1:y2, x1:x2]



#part 1: Spatial
# 	1.	Crop face
#	2.	Transform image
#	3.	Pass into ResNet
#	4.	Take softmax
#	5.	Return maximum probability

def spatial_stage(path: str, num_frames: int = 16) -> float:
    _check_path(path)

    def _score(rgb: np.ndarray) -> float:
        cropped = Image.fromarray(_crop_face(rgb))
        tensor = _resnet_transform(cropped).unsqueeze(0)
        with torch.no_grad():
            probs = torch.nn.functional.softmax(_resnet(tensor), dim=1)
        return float(torch.max(probs).item())

    if _is_image(path):
        return _score(np.array(Image.open(path).convert("RGB")))

    return float(np.mean([_score(f) for f in _load_frames(path, num_frames)]))


#part 2: Texture
#	1.	Crop face
#	2.	Convert to grayscale
#	3.	Resize to 128x128
#	4.	Compute Laplacian
#	5.	Compute variance
#	6.	Normalize by dividing by 2000

def texture_stage(path: str, num_frames: int = 16) -> float:
    _check_path(path)

    def _score(rgb: np.ndarray) -> float:
        cropped = _crop_face(rgb)
        gray = cv2.resize(cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY), (128, 128))
        return float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 2000.0, 0.0, 1.0))

    if _is_image(path):
        return _score(np.array(Image.open(path).convert("RGB")))

    return float(np.mean([_score(f) for f in _load_frames(path, num_frames)]))


#part 3: Motion
# 	1.	Load frames
#	2.	Crop face each frame
#	3.	Convert to grayscale
#	4.	Compute pixel difference between consecutive frames
#	5.	Normalize by 255
#	6.	Average differences

def motion_stage(path: str, num_frames: int = 16) -> float:
    _check_path(path)

    if _is_image(path):
        return 0.0

    frames = _load_frames(path, num_frames)
    if len(frames) < 2:
        return 0.0

    TARGET_SIZE = (128, 128)
    NORM_FACTOR = 255.0
    BOOST = 3.5  # calibration scalar — tune this

    diffs = []
    prev_crop = _crop_face(frames[0])
    prev = cv2.resize(
        cv2.cvtColor(prev_crop, cv2.COLOR_RGB2GRAY),
        TARGET_SIZE
    ).astype(np.float32)

    for f in frames[1:]:
        crop = _crop_face(f)
        cur = cv2.resize(
            cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY),
            TARGET_SIZE
        ).astype(np.float32)

        # edge map weights — focus on structural motion not noise
        edge_weight = cv2.Laplacian(prev, cv2.CV_32F)
        edge_weight = np.abs(edge_weight)
        edge_weight = edge_weight / (edge_weight.mean() + 1e-6)
        edge_weight = np.clip(edge_weight, 0.0, 3.0)

        diff = np.abs(cur - prev) / NORM_FACTOR
        weighted_diff = np.mean(diff * edge_weight) * BOOST

        diffs.append(weighted_diff)
        prev = cur

    if not diffs:
        return 0.0

    return float(np.clip(np.mean(diffs), 0.0, 1.0))


#part 4: Depth
# 	1.	Crop face
#	2.	Pass through depth model
#	3.	Upsample depth map to face size
#	4.	Compute variance of depth
#	5.	Normalize by dividing by 50000

def depth_stage(path: str, num_frames: int = 4) -> float:
    _check_path(path)
    _load_midas()

    frames = [np.array(Image.open(path).convert("RGB"))] if _is_image(path) \
             else _load_frames(path, num_frames)

    variances = []
    for frame in frames:
        cropped = _crop_face(frame)
        input_tensor = _midas_transform(cropped)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            raw_depth = _midas(input_tensor)
            depth = torch.nn.functional.interpolate(
                raw_depth.unsqueeze(1),
                size=cropped.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        variances.append(float(np.clip(depth.var().item() / 50000.0, 0.0, 1.0)))

    return float(np.mean(variances))


#diagnostics
def print_crop_stats() -> None:
    d = _crop_stats["detected"]
    f = _crop_stats["fallback"]
    total = d + f
    rate = 100.0 * d / total if total > 0 else 0.0
    print(f"[crop_stats] Face detected: {d}/{total} frames ({rate:.1f}%)  |  Fallback to full frame: {f}")


# Label rules 
#   live_selfie       =1  (live, static image — motion stage returns 0.0)
#   live_video        =1  (live, video, all four stages active)
#   cut-out printouts =0  (spoof: printed photo held up on video)
#   printouts         =0  (spoof: static printed photo)
#   replay            =0  (spoof: screen replay video)
def iter_dataset(root: str):
    samples = []
    for folder in sorted(os.listdir(root)):
        folder_path = os.path.join(root, folder)
        if not os.path.isdir(folder_path):
            continue
        label = 1 if folder.lower().startswith("live") else 0
        for file in sorted(os.listdir(folder_path)):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi")):
                samples.append((os.path.join(folder_path, file), label, folder))
    return samples