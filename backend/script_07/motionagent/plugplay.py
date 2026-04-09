import os
import time
import json
import glob
import numpy as np
import torch
from dotenv import load_dotenv
from openai import OpenAI

from models.motion_agent import MotionAgent
from options.option_llm import get_args_parser


class MotionAgentRunner:
    def __init__(self, device="cuda:0", save_dir="./demo"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        load_dotenv(os.path.join(base_dir, ".env"))

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")

        self.client = OpenAI(api_key=api_key)

        self.args = get_args_parser()
        self.args.device = device
        self.args.save_dir = save_dir

        self.agent = MotionAgent(self.args, self.client)

    def _get_latest_file(self, extension):
        pattern = os.path.join(self.agent.save_dir, f"*{extension}")
        files = glob.glob(pattern)
        if not files:
            return None
        return max(files, key=os.path.getmtime)

    def run(self, text, return_motion=True):
        before_mp4 = set(glob.glob(os.path.join(self.agent.save_dir, "*.mp4")))
        before_npy = set(glob.glob(os.path.join(self.agent.save_dir, "*.npy")))

        old_context_len = len(self.agent.context)
        old_motion_keys = set(self.agent.motion_history.keys())

        result = {
            "input_text": text,
            "assistant_response": None,
            "plan": None,
            "reasoning": None,
            "caption": None,
            "video_path": None,
            "motion_path": None,
            "motion_array": None,
            "motion_tokens": None,
            "new_context": None,
            "motion_history_keys": None,
            "error": None,
        }

        try:
            self.agent.process_motion_dialogue(text)

            if len(self.agent.context) > old_context_len:
                new_context = self.agent.context[old_context_len:]
                result["new_context"] = new_context

                for item in reversed(new_context):
                    if item["role"] == "assistant":
                        assistant_response = item["content"]
                        result["assistant_response"] = assistant_response
                        try:
                            parsed = json.loads(assistant_response)
                            result["plan"] = parsed.get("plan")
                            result["reasoning"] = parsed.get("reasoning")
                        except Exception:
                            pass
                        break

            after_mp4 = set(glob.glob(os.path.join(self.agent.save_dir, "*.mp4")))
            after_npy = set(glob.glob(os.path.join(self.agent.save_dir, "*.npy")))

            new_mp4 = sorted(after_mp4 - before_mp4, key=os.path.getmtime if after_mp4 - before_mp4 else lambda x: 0)
            new_npy = sorted(after_npy - before_npy, key=os.path.getmtime if after_npy - before_npy else lambda x: 0)

            if new_mp4:
                result["video_path"] = new_mp4[-1]
            else:
                result["video_path"] = self._get_latest_file(".mp4")

            if new_npy:
                result["motion_path"] = new_npy[-1]
            else:
                result["motion_path"] = self._get_latest_file(".npy")

            if return_motion and result["motion_path"] is not None:
                result["motion_array"] = np.load(result["motion_path"])

            new_motion_keys = list(set(self.agent.motion_history.keys()) - old_motion_keys)
            result["motion_history_keys"] = list(self.agent.motion_history.keys())

            if new_motion_keys:
                latest_key = new_motion_keys[-1]
                motion_tokens = self.agent.motion_history[latest_key]
                result["motion_tokens"] = motion_tokens
                result["caption"] = latest_key

            if result["plan"] and "caption" in result["plan"]:
                if self.agent.context:
                    for item in reversed(self.agent.context):
                        if item["role"] == "assistant":
                            try:
                                parsed = json.loads(item["content"])
                                if parsed.get("reasoning") is not None:
                                    result["reasoning"] = parsed.get("reasoning")
                                    break
                            except Exception:
                                continue

        except Exception as e:
            result["error"] = str(e)

        return result

    def clean(self):
        self.agent.clean()


if __name__ == "__main__":
    runner = MotionAgentRunner(device="cuda:0", save_dir="./demo")

    result = runner.run("A man is doing cartwheels.")

    print("=== RESULT KEYS ===")
    print(result.keys())

    print("\n=== ASSISTANT RESPONSE ===")
    print(result["assistant_response"])

    print("\n=== PLAN ===")
    print(result["plan"])

    print("\n=== REASONING ===")
    print(result["reasoning"])

    print("\n=== VIDEO PATH ===")
    print(result["video_path"])

    print("\n=== MOTION PATH ===")
    print(result["motion_path"])

    if result["motion_array"] is not None:
        print("\n=== MOTION ARRAY SHAPE ===")
        print(result["motion_array"].shape)

    if result["motion_tokens"] is not None:
        print("\n=== MOTION TOKENS SHAPE ===")
        try:
            print(result["motion_tokens"].shape)
        except Exception:
            print(type(result["motion_tokens"]))

    print("\n=== ERROR ===")
    print(result["error"])
"""
runner = MotionAgentRunner(device="cuda:0", save_dir="./demo")
result = runner.run("A person runs forward and then stops.")
print(result["motion_array"])
print(result["video_path"])

runner = MotionAgentRunner(device="cuda:0", save_dir="./demo")
result = runner.run("Describe this motion example.npy")
print(result["reasoning"])
print(result["assistant_response"])
"""