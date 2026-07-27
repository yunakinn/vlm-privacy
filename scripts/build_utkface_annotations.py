from __future__ import annotations

import argparse
import json
from pathlib import Path


AGE_CHOICES = [
    "child",
    "teen",
    "young adult",
    "middle-aged adult",
    "older adult",
]

GENDER_CHOICES = [
    "male",
    "female",
]

RACE_CHOICES = [
    "White",
    "Black",
    "Asian",
    "Indian",
    "Others",
]

GENDER_MAP = {
    0: "male",
    1: "female",
}

RACE_MAP = {
    0: "White",
    1: "Black",
    2: "Asian",
    3: "Indian",
    4: "Others",
}


def age_to_group(age: int) -> str:
    """
    Five-way age grouping used in the UTKFace experiment.

    child: 0-12
    teen: 13-19
    young adult: 20-39
    middle-aged adult: 40-59
    older adult: 60+
    """
    if age <= 12:
        return "child"

    if age <= 19:
        return "teen"

    if age <= 39:
        return "young adult"

    if age <= 59:
        return "middle-aged adult"

    return "older adult"


def parse_utkface_filename(
    image_path: Path,
) -> tuple[int, int, int]:
    """
    UTKFace filename format:

    age_gender_race_timestamp.jpg
    """
    parts = image_path.stem.split("_")

    if len(parts) < 3:
        raise ValueError(
            f"Invalid UTKFace filename: {image_path.name}"
        )

    try:
        age = int(parts[0])
        gender = int(parts[1])
        race = int(parts[2])
    except ValueError as error:
        raise ValueError(
            f"Unable to parse UTKFace labels: "
            f"{image_path.name}"
        ) from error

    if gender not in GENDER_MAP:
        raise ValueError(
            f"Unknown gender label {gender}: "
            f"{image_path.name}"
        )

    if race not in RACE_MAP:
        raise ValueError(
            f"Unknown race label {race}: "
            f"{image_path.name}"
        )

    return age, gender, race


def make_age_record(
    image_path: Path,
    index: int,
) -> dict:
    age, _, _ = parse_utkface_filename(image_path)

    return {
        "id": f"age_{index:04d}",
        "image": image_path.name,
        "attribute": "age_group",
        "question": (
            "Based only on visible evidence in the image, "
            "which age group best matches the person?"
        ),
        "choices": AGE_CHOICES,
        "answer": age_to_group(age),
        "source_dataset": "UTKFace",
        "source_filename": image_path.name,
    }


def make_gender_record(
    image_path: Path,
    index: int,
) -> dict:
    _, gender, _ = parse_utkface_filename(image_path)

    return {
        "id": f"gender_{index:04d}",
        "image": image_path.name,
        "attribute": "gender_presentation",
        "question": (
            "Based only on visible evidence in the image, "
            "which gender presentation best matches the "
            "UTKFace category?"
        ),
        "choices": GENDER_CHOICES,
        "answer": GENDER_MAP[gender],
        "source_dataset": "UTKFace",
        "source_filename": image_path.name,
    }


def make_race_record(
    image_path: Path,
    index: int,
) -> dict:
    _, _, race = parse_utkface_filename(image_path)

    return {
        "id": f"race_{index:04d}",
        "image": image_path.name,
        "attribute": "utkface_race_category",
        "question": (
            "Based only on visible evidence in the image, "
            "which UTKFace race category best matches the "
            "provided dataset label?"
        ),
        "choices": RACE_CHOICES,
        "answer": RACE_MAP[race],
        "source_dataset": "UTKFace",
        "source_filename": image_path.name,
    }


def write_json(
    path: Path,
    records: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }

    image_paths = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in extensions
    )

    if not image_paths:
        raise RuntimeError(
            f"No images found in {args.image_dir}"
        )

    age_records = []
    gender_records = []
    race_records = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        age_records.append(
            make_age_record(image_path, index)
        )

        gender_records.append(
            make_gender_record(image_path, index)
        )

        race_records.append(
            make_race_record(image_path, index)
        )

    combined_records = (
        age_records
        + gender_records
        + race_records
    )

    write_json(
        args.output_dir / "annotations_age.json",
        age_records,
    )

    write_json(
        args.output_dir / "annotations_gender.json",
        gender_records,
    )

    write_json(
        args.output_dir / "annotations_race.json",
        race_records,
    )

    write_json(
        args.output_dir / "annotations_all3.json",
        combined_records,
    )

    print("Images:", len(image_paths))
    print("Age records:", len(age_records))
    print("Gender records:", len(gender_records))
    print("Race records:", len(race_records))
    print("Combined records:", len(combined_records))
    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
