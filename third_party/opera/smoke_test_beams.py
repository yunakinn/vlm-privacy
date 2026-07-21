import os
import random

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

MODEL_EVAL_CONFIG_PATH = {"llava-1.5": "eval_configs/llava-1.5_eval.yaml"}
INSTRUCTION_TEMPLATE = {"llava-1.5": "USER: <ImageHere> <question> ASSISTANT:"}


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

IMAGE_PATH = "/content/smoke_test_img/test.jpg"
raw_image = Image.open(IMAGE_PATH).convert("RGB")
image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

qu = "Please describe this image in detail."
template = INSTRUCTION_TEMPLATE[args.model]
qu = template.replace("<question>", qu)

NUM_BEAMS = 5

with torch.inference_mode():
    with torch.no_grad():
        out1 = model.generate(
            {"image": norm(image), "prompt": qu},
            use_nucleus_sampling=False,
            num_beams=NUM_BEAMS,
            num_return_sequences=NUM_BEAMS,
            max_new_tokens=64,
            output_attentions=True,
            return_dict_in_generate=True,
            output_scores=True,
            opera_decoding=True,
            scale_factor=50,
            threshold=15.0,
            num_attn_candidates=5,
        )

output_text, raw_output = out1
print("num returned sequences:", len(output_text))
for i, t in enumerate(output_text):
    print(f"  beam {i}: {t[:80]}")
print("sequences_scores:", raw_output.sequences_scores)
print("rollback_count:", getattr(raw_output, "rollback_count", "NOT FOUND"))
