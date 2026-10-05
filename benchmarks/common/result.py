from pathlib import Path
import csv

class BenchmarkResults:
    def __init__(self, name: str):
        self.name = name

        self.results_dir = (
            Path(__file__).parent.parent / "results"
        )

        self.results_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.csv_file = (
            self.results_dir / f"{name}.csv"
        )

        self.md_file = (
            self.results_dir / f"{name}.md"
        )

    def create_csv(self, headers):
        with open(
            self.csv_file,
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(headers)

    def append_csv(self, row):
        with open(
            self.csv_file,
            "a",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(row)

    def append_markdown(self, text):
        with open(
            self.md_file,
            "a",
        ) as f:
            f.write(text + "\n")
