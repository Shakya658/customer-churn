"""Embed generated evaluation screenshots in the README once."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README_PATH = ROOT / "README.md"

SECTION = '''## Evaluation Screenshots

### Real held-out ROC curves

![Real held-out ROC curves](assets/screenshots/real_roc_curves.png)

### Real held-out precision-recall curves

![Real held-out precision-recall curves](assets/screenshots/real_precision_recall_curves.png)

'''


def main() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    if "## Evaluation Screenshots\n" not in readme:
        marker = "## Limitations\n"
        if marker not in readme:
            raise ValueError("README insertion marker not found")
        readme = readme.replace(marker, SECTION + marker, 1)
        README_PATH.write_text(readme, encoding="utf-8")
        print("Embedded evaluation screenshots in README")
    else:
        print("README screenshot section already present")


if __name__ == "__main__":
    main()
