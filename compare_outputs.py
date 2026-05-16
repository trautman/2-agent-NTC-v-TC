# compare_outputs.py

from pathlib import Path
import hashlib
import csv
import math
from PIL import Image
import numpy as np


BEFORE_DIR = Path(
    "/home/zed-user/Desktop/simulation/NTC_vs_TC_sim/two_agents_head_on/ntc_tc_sim_outputs_v20/results-before-refactor"
)

AFTER_DIR = Path(
    "/home/zed-user/Desktop/simulation/NTC_vs_TC_sim/two_agents_head_on/ntc_tc_sim_outputs_v20/results-after-refactor"
)

DIFFERENCES = []

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def is_float(x):
    try:
        float(x)
        return True
    except:
        return False


def compare_csv(before_path, after_path):
    print(f"\n[CSV] {before_path.name}")

    with open(before_path, newline="") as f:
        old_rows = list(csv.reader(f))

    with open(after_path, newline="") as f:
        new_rows = list(csv.reader(f))

    if len(old_rows) != len(new_rows):
        print("  ROW COUNT MISMATCH")
        print(f"  before: {len(old_rows)}")
        print(f"  after : {len(new_rows)}")
        return False

    if old_rows[0] != new_rows[0]:
        # print("  HEADER MISMATCH")
        msg = f"{before_path.name}: HEADER MISMATCH"
        print(f"  {msg}")
        DIFFERENCES.append(msg)
        return False

    for r in range(1, len(old_rows)):
        old = old_rows[r]
        new = new_rows[r]

        if len(old) != len(new):
            print(f"  COLUMN COUNT MISMATCH at row {r}")
            return False

        for c in range(len(old)):

            a = old[c]
            b = new[c]

            if is_float(a) and is_float(b):

                fa = float(a)
                fb = float(b)

                if math.isnan(fa) and math.isnan(fb):
                    continue

                if not math.isclose(fa, fb, rel_tol=1e-12, abs_tol=1e-12):
                    msg = (
                        f"{before_path.name}: NUMERIC MISMATCH "
                        f"(row={r}, col={c}, before={fa}, after={fb})"
                    )

                    print(f"  {msg}")
                    DIFFERENCES.append(msg)
                    return False

            else:
                if a != b:
                    msg = (
                        f"{before_path.name}: STRING MISMATCH "
                        f"(row={r}, col={c}, before={a}, after={b})"
                    )

                    print(f"  {msg}")
                    DIFFERENCES.append(msg)
                    return False

    print("  MATCH")
    return True


def compare_png(before_path, after_path):
    print(f"\n[PNG] {before_path.name}")

    a = np.array(Image.open(before_path))
    b = np.array(Image.open(after_path))

    if a.shape != b.shape:
        print("  IMAGE SHAPE MISMATCH")
        print(f"  before: {a.shape}")
        print(f"  after : {b.shape}")
        return False

    equal = np.array_equal(a, b)

    if not equal:
        num_diff = np.count_nonzero(a != b)
        # print(f"  PIXEL MISMATCH | differing values = {num_diff}")
        # return False
        msg = (
            f"{before_path.name}: PIXEL MISMATCH "
            f"(differing values={num_diff})"
        )

        print(f"  {msg}")
        DIFFERENCES.append(msg)
        return False

    print("  MATCH")
    return True


def compare_binary(before_path, after_path):
    print(f"\n[BINARY] {before_path.name}")

    h1 = sha256(before_path)
    h2 = sha256(after_path)

    if h1 != h2:
        # print("  HASH MISMATCH")
        # return False
        msg = f"{before_path.name}: HASH MISMATCH"

        print(f"  {msg}")
        DIFFERENCES.append(msg)
        return False

    print("  MATCH")
    return True


def main():
    before_files = sorted([p for p in BEFORE_DIR.iterdir() if p.is_file()])
    after_lookup = {p.name: p for p in AFTER_DIR.iterdir() if p.is_file()}

    all_ok = True

    before_names = {p.name for p in before_files}
    after_names = set(after_lookup.keys())

    extra_after = sorted(after_names - before_names)

    for name in extra_after:
        msg = f"{name}: EXTRA FILE in after-refactor"
        print(f"\n[EXTRA] {msg}")
        DIFFERENCES.append(msg)
        all_ok = False

    for before_path in before_files:

        name = before_path.name

        # if name not in after_lookup:
        #     print(f"\n[MISSING] {name} not found in after-refactor")
        #     all_ok = False
        #     continue
        if name not in after_lookup:
            msg = f"{name}: MISSING from after-refactor"
            print(f"\n[MISSING] {msg}")
            DIFFERENCES.append(msg)
            all_ok = False
            continue




        

        after_path = after_lookup[name]

        suffix = before_path.suffix.lower()

        try:
            if suffix == ".csv":
                ok = compare_csv(before_path, after_path)

            elif suffix == ".png":
                ok = compare_png(before_path, after_path)

            else:
                ok = compare_binary(before_path, after_path)

        except Exception as e:
            print(f"\n[ERROR] {name}")
            print(e)
            ok = False

        all_ok = all_ok and ok

    print("\n" + "=" * 60)

    # if all_ok:
    #     print("ALL FILES MATCH")
    # else:
    #     print("DIFFERENCES DETECTED")
    if all_ok:
        print("ALL FILES MATCH")
    else:
        print("DIFFERENCES DETECTED\n")

        for d in DIFFERENCES:
            print(d)

    print("=" * 60)


if __name__ == "__main__":
    main()