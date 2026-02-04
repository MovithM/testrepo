import os
import subprocess
import pandas as pd
import tempfile

# -------- CONFIG (FIXED) --------
RAVDESS_DIR = "/home/movith/Datasets/RAVDESS"
OPENSMILE_BIN = "/home/movith/opensmile/build/progsrc/smilextract/SMILExtract"
EGEMAPS_CONF = "/home/movith/opensmile/config/egemaps/v02/eGeMAPSv02.conf"

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

rows = []

# -------- PROCESS FILES --------
for root, _, files in os.walk(RAVDESS_DIR):
    for file in files:
        if not file.endswith(".wav"):
            continue

        filepath = os.path.join(root, file)
        print("Processing file:", filepath)


        # Emotion code from filename
        parts = file.split("-")
        if len(parts) < 3:
            continue

        emotion_code = parts[2]
        emotion = emotion_map.get(emotion_code)
        if emotion is None:
            continue

        # Temporary ARFF output
        tmp = tempfile.NamedTemporaryFile(suffix=".arff", delete=False)
        tmp.close()  # IMPORTANT: close before openSMILE writes
        cmd = [
            OPENSMILE_BIN,
            "-C", EGEMAPS_CONF,
            "-I", filepath,
            "-O", tmp.name
         ]

        result = subprocess.run(
            cmd,
            cwd="/home/movith/opensmile",
            capture_output=True,
            text=True
        )

        print("CMD:", " ".join(cmd))
        print("RETURNCODE:", result.returncode)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)



        # Parse ARFF
        with open(tmp.name, "r") as f:
            for line in f:
                line = line.strip()

                # Skip headers and empty lines
                if (
                    not line
                    or line.startswith("@")
                    or line.startswith("%")
                ):
                    continue

                values = line.split(",")

                # Expect: name + features + class
                if len(values) < 10:
                    continue

                features = values[1:-1]   # drop "name" and "class"
                rows.append(features + [emotion])

                print("✅ Feature row extracted")
                break




# -------- CREATE DATAFRAME --------
num_features = len(rows[0]) - 1
columns = [f"f{i}" for i in range(num_features)] + ["emotion"]

df = pd.DataFrame(rows, columns=columns)
df.to_csv("ravdess_egemaps.csv", index=False)

print("✅ Feature extraction complete!")
print(f"Total samples: {len(df)}")
print(df.head())
