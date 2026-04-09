from plugplay import MotionGPTRunner

def run_text_to_motion():

    runner = MotionGPTRunner()

    result = runner.run(
        text="A person runs forward and then stops.",
        task="t2m",
        render=True,
    )

    print("Generated motion result:")
    print(result)


def run_motion_to_text():

    runner = MotionGPTRunner()

    result = runner.run(
        text="Describe <Motion_Placeholder>.",
        task="m2t",
        motion_file="example.npy",
        render=False,
    )

    print("Generated description:")
    print(result["texts"])


if __name__ == "__main__":

    print("Running MotionGPT tasks...\n")

    run_text_to_motion()

    print("\n------------------------\n")

    run_motion_to_text()