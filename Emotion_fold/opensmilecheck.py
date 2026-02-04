import subprocess
import os

cmd = [
    "./SMILExtract",
    "-C", "../../../config/egemaps/v02/eGeMAPSv02.conf",
    "-I", "/home/movith/Datasets/RAVDESS/Actor_01/03-01-05-01-01-02-01.wav",
    "-O", "sanity_output.csv"
]

result = subprocess.run(
    cmd,
    cwd="/home/movith/opensmile/build/progsrc/smilextract",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

print("STDERR:\n", result.stderr)
print("\n--- OUTPUT FILE CONTENT ---\n")

with open("/home/movith/opensmile/build/progsrc/smilextract/sanity_output.csv") as f:
    print(f.read())
