"""SafeLIBERO client running original OpenVLA-7B for π₀.₅-style baseline rollouts.

Single-process: loads OpenVLA from HuggingFace and steps SafeLIBERO env directly.
No websocket server needed.

Usage:
    export PYTHONPATH=$PYTHONPATH:vlsa-aegis/safelibero
    python main_demo_openvla.py --task-suite-name safelibero_spatial --safety-level I \
        --task-index 0 --episode-index 0 --video-out-path data/libero/videos_openvla
"""
import dataclasses
import logging
import pathlib
from typing import List

import imageio
import numpy as np
import torch
import tqdm
import tyro
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    # Model
    model_id: str = "openvla/openvla-7b-finetuned-libero-spatial"
    unnorm_key: str = "libero_spatial_no_noops"
    device: str = "cuda:0"

    # SafeLIBERO
    task_suite_name: str = "safelibero_spatial"
    safety_level: str = "I"
    task_index: List[int] = dataclasses.field(default_factory=lambda: [0])
    episode_index: List[int] = dataclasses.field(default_factory=lambda: [0])
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    video_out_path: str = "data/libero/videos_openvla"
    seed: int = 7


def _resize_image(img_np: np.ndarray, size: int = 224) -> Image.Image:
    """Rotate 180° (LIBERO convention) and resize to model input size."""
    img = np.ascontiguousarray(img_np[::-1, ::-1])
    pil = Image.fromarray(img).convert("RGB").resize((size, size), Image.LANCZOS)
    return pil


def _normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """OpenVLA outputs gripper in [0, 1] dataset normalization. LIBERO wants
    {-1, +1}. Rescale [0,1] -> [-1, +1], optionally binarize."""
    action = action.copy()
    action[..., -1] = 2 * (action[..., -1] - 0.5)
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """OpenVLA convention: 0=close, 1=open. LIBERO convention: -1=open, +1=close.
    After normalize_gripper_action -> {-1, +1}, invert sign."""
    action = action.copy()
    action[..., -1] = -action[..., -1]
    return action


def _get_libero_env(task, level, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)

    # Load model + processor using prismatic's loader (bypasses HF trust_remote_code)
    logging.info(f"Loading OpenVLA: {args.model_id}")
    from transformers import AutoConfig, AutoImageProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(args.device)
    vla.eval()
    logging.info("Model loaded.")

    # Setup SafeLIBERO
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name](safety_level=args.safety_level)
    logging.info(f"Task suite: {args.task_suite_name}, safety level: {args.safety_level}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name in ("safelibero_spatial", "safelibero_object", "safelibero_goal"):
        max_steps = 300
    elif args.task_suite_name == "safelibero_long":
        max_steps = 500
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(args.task_index):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, args.safety_level, LIBERO_ENV_RESOLUTION, args.seed)
        prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(args.episode_index):
            logging.info(f"\nTask: {task_description}")
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            done = False

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Build inputs
                    pil_img = _resize_image(obs["agentview_image"], size=224)
                    replay_images.append(np.array(pil_img))
                    inputs = processor(prompt, pil_img).to(args.device, dtype=torch.bfloat16)

                    # Inference
                    with torch.no_grad():
                        action = vla.predict_action(**inputs, unnorm_key=args.unnorm_key, do_sample=False)
                    action = np.asarray(action)

                    # Convert gripper from OpenVLA's [0,1] convention to LIBERO's {-1,+1}
                    action = _normalize_gripper_action(action, binarize=True)
                    action = _invert_gripper_action(action)

                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            video_path = pathlib.Path(args.video_out_path) / \
                f"rollout_{task_segment}_{args.safety_level}_{episode_idx}_{suffix}.mp4"
            imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
            logging.info(f"Saved video to {video_path}")
            logging.info(f"Success: {done}")
            logging.info(f"# episodes: {total_episodes}, # successes: {total_successes} "
                         f"({total_successes / total_episodes * 100:.1f}%)")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    eval_libero(args)