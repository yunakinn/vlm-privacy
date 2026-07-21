import json
import os
import random
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from torchvision import transforms

from minigpt4.common.dist_utils import get_rank
from minigpt4.models import load_preprocess
from minigpt4.common.config import Config
from minigpt4.common.registry import registry

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image

from opera_choice_scoring import run_opera_choice

MODEL_EVAL_CONFIG_PATH = {"llava-1.5": "eval_configs/llava-1.5_eval.yaml"}


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


class Args:
    model = "llava-1.5"
    gpu_id = "0"
    batch_size = 1
    options = None


args = Args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
args.cfg_path = MODEL_EVAL_CONFIG_PATH[args.model]
cfg = Config(args)
setup_seeds(cfg)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

print("Initializing Model")
model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
model.eval()
processor_cfg = cfg.get_config().preprocess
processor_cfg.vis_processor.eval.do_normalize = False
vis_processors, txt_processors = load_preprocess(processor_cfg)
print("Model loaded!")

mean = (0.48145466, 0.4578275, 0.40821073)
std = (0.26862954, 0.26130258, 0.27577711)
norm = transforms.Normalize(mean, std)

DATA_DIR = "/content/drive/MyDrive/vlm-privacy/datasets/utkface_balanced30_seed42"
with open(f"{DATA_DIR}/annotations_age_30.json") as f:
    annotations = json.load(f)

results = []
start = time.time()

for i, sample in enumerate(annotations):
    image_path = f"{DATA_DIR}/images/{sample['image']}"
    image = Image.open(image_path)

    result = run_opera_choice(
        model=model,
        vis_processors=vis_processors,
        norm_fn=norm,
        image=image,
        question=sample["question"],
        choices=sample["choices"],
        num_beams=5,
        max_new_tokens=8,
        device=device,
    )
    result["id"] = sample["id"]
    result["ground_truth"] = sample["answer"]
    result["correct"] = (result["parsed_label"] == sample["answer"])
    results.append(result)

    elapsed = time.time() - start
    print(
        f"[{i+1}/30] {sample['id']}: pred={result['prediction']} "
        f"({result['parsed_label']}) gt={sample['answer']} "
        f"correct={result['correct']} rollback={result['rollback_count']} "
        f"margin={result['beam_margin']:.4f} "
        f"changed={result['opera_changed_prediction']} "
        f"[{elapsed:.0f}s elapsed]"
    )

# Drop bulky per-beam debug fields before saving
save_results = []
for r in results:
    r_save = {k: v for k, v in r.items() if k not in ("opera_beams", "plain_beams")}
    save_results.append(r_save)

os.makedirs("/content/vlm-privacy/results/opera_pilot30_age", exist_ok=True)
out_path = "/content/vlm-privacy/results/opera_pilot30_age/results.json"
with open(out_path, "w") as f:
    json.dump(save_results, f, indent=2)

n_correct = sum(r["correct"] for r in results)
n_valid = sum(r["is_valid"] for r in results)
n_rollback = sum(1 for r in results if r["rollback_count"] > 0)
n_changed = sum(r["opera_changed_prediction"] for r in results)

print()
print("=== 요약 ===")
print(f"accuracy: {n_correct}/30 = {n_correct/30:.4f}")
print(f"valid parses: {n_valid}/30")
print(f"samples with rollback_count > 0: {n_rollback}/30")
print(f"samples with opera_changed_prediction: {n_changed}/30")
print(f"total time: {time.time()-start:.0f}s")
print(f"saved to: {out_path}")
