#!/usr/bin/env python3
"""Create fixed MVUB train and held-out Andrew validation manifests."""

import argparse
import json
import os
import re


TRAIN_SUBJECTS = ("David10", "Phil10", "Ricardo10", "Sarah10")
VALIDATION_SUBJECT = "Andrew10"
VALIDATION_FRAMES = (
    "frame0000", "frame0045", "frame0091", "frame0136",
    "frame0181", "frame0226", "frame0272", "frame0317",
)
FRAME_PATTERN = re.compile(
    r"^(?P<subject>[A-Za-z0-9]+)_(?P<frame>frame[0-9]{4})_P[0-9]{3}\.h5$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def subject_h5(root, subject):
    directory = os.path.join(root, subject)
    if not os.path.isdir(directory):
        raise FileNotFoundError("MVUB subject directory missing: " + directory)
    rows = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".h5"):
            continue
        match = FRAME_PATTERN.match(name)
        if match is None or match.group("subject") != subject:
            raise ValueError("Unexpected MVUB H5 filename: " + name)
        rows.append((match.group("frame"), os.path.join(subject, name)))
    if not rows:
        raise RuntimeError("No H5 files for MVUB subject " + subject)
    return rows


def write_lines(path, values):
    with open(path, "w", encoding="utf-8") as handle:
        handle.writelines(value + "\n" for value in values)


def main():
    args = parse_args()
    root = os.path.abspath(args.dataset_root)
    output = os.path.abspath(args.output_dir)
    os.makedirs(output, exist_ok=True)
    protected = ("train_h5.txt", "andrew_val8_h5.txt", "manifest_info.json")
    existing = [name for name in protected
                if os.path.exists(os.path.join(output, name))]
    if existing:
        raise FileExistsError("Refusing to overwrite: " + ", ".join(existing))

    train = []
    train_counts = {}
    for subject in TRAIN_SUBJECTS:
        rows = subject_h5(root, subject)
        train.extend(relative for _, relative in rows)
        train_counts[subject] = len(rows)

    andrew = subject_h5(root, VALIDATION_SUBJECT)
    available_frames = sorted(set(frame for frame, _ in andrew))
    missing_frames = [frame for frame in VALIDATION_FRAMES
                      if frame not in available_frames]
    if missing_frames:
        raise ValueError("Andrew validation frames missing: " + ", ".join(
            missing_frames))
    validation = [relative for frame, relative in andrew
                  if frame in VALIDATION_FRAMES]
    if set(train) & set(validation):
        raise RuntimeError("MVUB train/validation H5 overlap")

    write_lines(os.path.join(output, "train_h5.txt"), train)
    write_lines(os.path.join(output, "andrew_val8_h5.txt"), validation)
    info = {
        "status": "PASS",
        "dataset_root_at_generation": root,
        "paths": "relative_to_dataset_root",
        "train_subjects": list(TRAIN_SUBJECTS),
        "train_h5_by_subject": train_counts,
        "train_h5": len(train),
        "validation_subject": VALIDATION_SUBJECT,
        "validation_selection": "8 uniformly spaced fixed frames",
        "validation_frames": list(VALIDATION_FRAMES),
        "validation_h5": len(validation),
        "overlap": 0,
    }
    with open(os.path.join(output, "manifest_info.json"), "w",
              encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
