from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


DATASETS: Dict[str, Dict[str, Any]] = {
    "1": {
        "name": "Devign",
        "files": [
            Path(r"d:\Projects\SemVul\explanations\SemanticVul\ACTIVE\devign\devign_final_train_3.jsonl"),
            Path(r"d:\Projects\SemVul\explanations\SemanticVul\ACTIVE\devign\devign_final_val_3.jsonl"),
        ],
    },
    "2": {
        "name": "Reveal",
        "files": [
            Path(r"d:\Projects\SemVul\explanations\SemanticVul\ACTIVE\reveal\reveal_final_train_3.jsonl"),
            Path(r"d:\Projects\SemVul\explanations\SemanticVul\ACTIVE\reveal\reveal_final_val_3.jsonl"),
        ],
    },
}

SIGNAL_STRENGTHS: Dict[str, str] = {
    "1": "very_weak",
    "2": "weak",
    "3": "medium",
    "4": "strong",
}

SIGNAL_LABELS: Dict[str, str] = {
    "very_weak": "Very Weak",
    "weak": "Weak",
    "medium": "Medium",
    "strong": "Strong",
}

PERCENTAGES: Dict[str, int] = {
    "1": 20,
    "2": 25,
    "3": 30,
    "4": 35,
    "5": 40,
    "6": 45,
    "7": 50,
    "8": 55,
    "9": 60,
    "10": 65,
    "11": 70,
    "12": 72,
    "13": 75,
    "14": 80,
    "15": 85,
    "16": 90,
    "17": 95,
    "18": 100,
}

RISK_RANGES: Dict[str, Tuple[int, int]] = {
    "LOW": (15, 45),
    "MEDIUM": (35, 70),
    "HIGH": (60, 98),
}

# Each signal strength defines how strongly risk metadata follows the label.
# We keep separate distributions for:
# - selected_positive: label==1 rows chosen by the requested percentage
# - positive: label==1 rows not chosen by that percentage
# - negative: label==0 rows
RISK_WEIGHTS: Dict[str, Dict[str, Dict[str, int]]] = {
    "very_weak": {
        "selected_positive": {"LOW": 25, "MEDIUM": 45, "HIGH": 30},
        "positive": {"LOW": 35, "MEDIUM": 40, "HIGH": 25},
        "negative": {"LOW": 25, "MEDIUM": 35, "HIGH": 40},
    },
    "weak": {
        "selected_positive": {"LOW": 15, "MEDIUM": 35, "HIGH": 50},
        "positive": {"LOW": 25, "MEDIUM": 45, "HIGH": 30},
        "negative": {"LOW": 45, "MEDIUM": 35, "HIGH": 20},
    },
    "medium": {
        "selected_positive": {"LOW": 8, "MEDIUM": 22, "HIGH": 70},
        "positive": {"LOW": 18, "MEDIUM": 40, "HIGH": 42},
        "negative": {"LOW": 62, "MEDIUM": 28, "HIGH": 10},
    },
    "strong": {
        "selected_positive": {"LOW": 2, "MEDIUM": 8, "HIGH": 90},
        "positive": {"LOW": 12, "MEDIUM": 28, "HIGH": 60},
        "negative": {"LOW": 78, "MEDIUM": 18, "HIGH": 4},
    },
}

# Confidence values are sampled from overlapping ranges using triangular
# distributions so they look varied instead of deterministic.
CONFIDENCE_MODES: Dict[str, Dict[str, Dict[str, int]]] = {
    "very_weak": {
        "selected_positive": {"LOW": 28, "MEDIUM": 48, "HIGH": 72},
        "positive": {"LOW": 26, "MEDIUM": 46, "HIGH": 70},
        "negative": {"LOW": 30, "MEDIUM": 48, "HIGH": 68},
    },
    "weak": {
        "selected_positive": {"LOW": 24, "MEDIUM": 55, "HIGH": 80},
        "positive": {"LOW": 24, "MEDIUM": 50, "HIGH": 76},
        "negative": {"LOW": 28, "MEDIUM": 45, "HIGH": 70},
    },
    "medium": {
        "selected_positive": {"LOW": 22, "MEDIUM": 58, "HIGH": 90},
        "positive": {"LOW": 22, "MEDIUM": 54, "HIGH": 84},
        "negative": {"LOW": 24, "MEDIUM": 42, "HIGH": 66},
    },
    "strong": {
        "selected_positive": {"LOW": 20, "MEDIUM": 62, "HIGH": 96},
        "positive": {"LOW": 20, "MEDIUM": 58, "HIGH": 90},
        "negative": {"LOW": 20, "MEDIUM": 40, "HIGH": 63},
    },
}


def prompt_menu(title: str, options: Sequence[Tuple[str, str]]) -> str:
    """Display a numbered menu until the user enters a valid choice."""
    while True:
        print(title)
        for key, label in options:
            print(f"{key}. {label}")

        choice = input("Enter choice: ").strip()
        if choice in dict(options):
            return choice

        print("Invalid choice. Please try again.\n")


def choose_dataset() -> Dict[str, Any]:
    dataset_choice = prompt_menu(
        "Select dataset:",
        [("1", "Devign"), ("2", "Reveal")],
    )
    return DATASETS[dataset_choice]


def choose_signal_strength() -> str:
    strength_choice = prompt_menu(
        "\nSelect signal strength:",
        [
            ("1", "Very Weak"),
            ("2", "Weak"),
            ("3", "Medium"),
            ("4", "Strong"),
        ],
    )
    return SIGNAL_STRENGTHS[strength_choice]


def choose_percentage() -> int:
    percent_choice = prompt_menu(
        "\nSelect percentage of label==1 rows to receive the intended signal:",
        [
            ("1", "20%"),
            ("2", "25%"),
            ("3", "30%"),
            ("4", "35%"),
            ("5", "40%"),
            ("6", "45%"),
            ("7", "50%"),
            ("8", "55%"),
            ("9", "60%"),
            ("10", "65%"),
            ("11", "70%"),
            ("12", "72%"),
            ("13", "75%"),
            ("14", "80%"),
            ("15", "85%"),
            ("16", "90%"),
            ("17", "95%"),
            ("18", "100%"),
        ],
    )
    return PERCENTAGES[percent_choice]


def load_records(file_path: Path) -> Tuple[str, List[Dict[str, Any]], List[str]]:
    """Load records from JSONL or CSV while preserving other fields."""
    suffix = file_path.suffix.lower()
    if suffix == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_number} in {file_path}"
                    ) from exc
        return "jsonl", rows, []

    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            return "csv", rows, list(reader.fieldnames or [])

    raise ValueError(f"Unsupported file type: {file_path}")


def save_records(
    file_path: Path,
    file_type: str,
    rows: Sequence[Dict[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    """Overwrite the original file with updated records."""
    if file_type == ".jsonl":
        raise ValueError("Unexpected file type sentinel '.jsonl'.")

    if file_type == "jsonl":
        with file_path.open("w", encoding="utf-8", newline="") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return

    if file_type == "csv":
        output_fields = list(fieldnames)
        for extra in ("risk_level", "confidence"):
            if extra not in output_fields:
                output_fields.append(extra)

        with file_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(rows)
        return

    raise ValueError(f"Unsupported file type: {file_type}")


def parse_label(row: Dict[str, Any], file_path: Path) -> int:
    """Return the integer label and validate that it is 0 or 1."""
    if "label" not in row:
        raise KeyError(f"Missing 'label' field in {file_path}")

    try:
        label = int(row["label"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid label value in {file_path}: {row['label']}") from exc

    if label not in (0, 1):
        raise ValueError(f"Label must be 0 or 1 in {file_path}, got {label}")

    return label


def get_signal_container(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update nested explanation metadata when it exists.
    Fall back to the top-level record for flat CSV-style rows.
    """
    explanation = row.get("explanation")
    if isinstance(explanation, dict):
        return explanation
    return row


def clear_signal_fields(rows: Iterable[Dict[str, Any]]) -> None:
    """Blank old values before regenerating them."""
    for row in rows:
        target = get_signal_container(row)
        target["risk_level"] = ""
        target["confidence"] = ""


def choose_selected_positive_indices(
    rows: Sequence[Dict[str, Any]],
    file_path: Path,
    percentage: int,
) -> set[int]:
    """Select the fraction of positive rows that receive the intended signal."""
    positive_indices = [
        index
        for index, row in enumerate(rows)
        if parse_label(row, file_path) == 1
    ]
    selected_count = round(len(positive_indices) * (percentage / 100.0))
    if selected_count == 0:
        return set()
    return set(random.sample(positive_indices, selected_count))


def sample_risk_level(strength: str, cohort: str) -> str:
    """Sample LOW / MEDIUM / HIGH using weighted probabilities."""
    weights = RISK_WEIGHTS[strength][cohort]
    levels = list(weights.keys())
    probabilities = list(weights.values())
    return random.choices(levels, weights=probabilities, k=1)[0]


def sample_confidence_value(strength: str, cohort: str, risk_level: str) -> int:
    """Sample a realistic confidence score with overlap across categories."""
    lower, upper = RISK_RANGES[risk_level]
    mode = CONFIDENCE_MODES[strength][cohort][risk_level]
    sampled = int(round(random.triangular(lower, upper, mode)))
    return max(0, min(100, sampled))


def assign_signal_values(
    rows: Sequence[Dict[str, Any]],
    file_path: Path,
    strength: str,
    percentage: int,
) -> Dict[str, int]:
    """Clear existing values, then regenerate them row by row."""
    clear_signal_fields(rows)
    selected_positive_indices = choose_selected_positive_indices(rows, file_path, percentage)

    stats = {
        "rows": 0,
        "positive_rows": 0,
        "selected_positive_rows": len(selected_positive_indices),
        "negative_rows": 0,
    }

    for index, row in enumerate(rows):
        label = parse_label(row, file_path)
        if label == 1:
            stats["positive_rows"] += 1
            cohort = "selected_positive" if index in selected_positive_indices else "positive"
        else:
            stats["negative_rows"] += 1
            cohort = "negative"

        risk_level = sample_risk_level(strength, cohort)
        confidence = sample_confidence_value(strength, cohort, risk_level)

        target = get_signal_container(row)
        target["risk_level"] = risk_level
        target["confidence"] = confidence
        stats["rows"] += 1

    return stats


def process_file(file_path: Path, strength: str, percentage: int) -> Dict[str, int]:
    """Load one dataset file, rewrite the generated metadata, and save it."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    file_type, rows, fieldnames = load_records(file_path)
    stats = assign_signal_values(rows, file_path, strength, percentage)
    save_records(file_path, file_type, rows, fieldnames)
    return stats


def main() -> None:
    dataset = choose_dataset()
    strength = choose_signal_strength()
    percentage = choose_percentage()

    print(
        f"\nProcessing {dataset['name']} with {SIGNAL_LABELS[strength]} signal "
        f"and {percentage}% selected positives..."
    )

    total_rows = 0
    total_positive = 0
    total_selected_positive = 0

    for file_path in dataset["files"]:
        stats = process_file(file_path, strength, percentage)
        total_rows += stats["rows"]
        total_positive += stats["positive_rows"]
        total_selected_positive += stats["selected_positive_rows"]
        print(
            f"Updated {file_path} | rows={stats['rows']} "
            f"positives={stats['positive_rows']} "
            f"selected_positives={stats['selected_positive_rows']}"
        )

    print(
        f"\nDone. Updated {len(dataset['files'])} files, {total_rows} rows total, "
        f"with {total_selected_positive}/{total_positive} positive rows receiving "
        f"the intended signal."
    )


if __name__ == "__main__":
    main()
