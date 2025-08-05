import sys, os

prompt = ""
dir_path = os.path.abspath(os.path.dirname(__file__))
filename_prompt = "prompt_3.txt"

settings = {
    "filepath": dir_path + os.path.sep + filename_prompt,
    "mode": "r"
}

def get_prompt():
    with open(settings["filepath"], settings["mode"]) as f:
        prompt = f.read()

    return prompt

if __name__ == "__main__":
    print("test: ", get_prompt())

