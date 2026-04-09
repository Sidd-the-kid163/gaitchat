import argparse
from motionagent_runner import MotionAgentRunner


def run_text_to_motion(text, device, save_dir):

    runner = MotionAgentRunner(
        device=device,
        save_dir=save_dir
    )

    result = runner.run(text)

    print("Motion Array:")
    print(result["motion_array"])

    print("\nVideo Path:")
    print(result["video_path"])


def run_motion_to_text(text, device, save_dir):

    runner = MotionAgentRunner(
        device=device,
        save_dir=save_dir
    )

    result = runner.run(text)

    print("\nReasoning:")
    print(result["reasoning"])

    print("\nAssistant Response:")
    print(result["assistant_response"])


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Prompt text"
    )

    parser.add_argument(
        "--task",
        type=str,
        choices=["t2m", "m2t"],
        required=True,
        help="Task type"
    )

    parser.add_argument(
        "--device",
        default="cuda:0"
    )

    parser.add_argument(
        "--save_dir",
        default="./demo"
    )

    args = parser.parse_args()

    if args.task == "t2m":
        run_text_to_motion(args.text, args.device, args.save_dir)

    elif args.task == "m2t":
        run_motion_to_text(args.text, args.device, args.save_dir)


if __name__ == "__main__":
    main()