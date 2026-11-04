"""Block commits containing IBANs, US bank numbers, or personal terms."""

import argparse
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

SENSITIVE_TERMS_FILE = Path(".sensitive-terms")
ALLOW_MARKER = "sensitive-data: allow"

# Total IBAN length per country, from the SWIFT IBAN Registry.
IBAN_LENGTHS = {
    "AD": 24, "AE": 23, "AL": 28, "AT": 20, "AZ": 28, "BA": 20, "BE": 16,
    "BG": 22, "BH": 22, "BR": 29, "BY": 28, "CH": 21, "CR": 22, "CY": 28,
    "CZ": 24, "DE": 22, "DK": 18, "DO": 28, "EE": 20, "EG": 29, "ES": 24,
    "FI": 18, "FO": 18, "FR": 27, "GB": 22, "GE": 22, "GI": 23, "GL": 18,
    "GR": 27, "GT": 28, "HR": 21, "HU": 28, "IE": 22, "IL": 23, "IQ": 23,
    "IS": 26, "IT": 27, "JO": 30, "KW": 30, "KZ": 20, "LB": 28, "LC": 32,
    "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MC": 27, "MD": 24, "ME": 22,
    "MK": 19, "MR": 27, "MT": 31, "MU": 30, "NL": 18, "NO": 15, "PK": 24,
    "PL": 28, "PS": 29, "PT": 25, "QA": 29, "RO": 24, "RS": 22, "SA": 24,
    "SC": 31, "SE": 24, "SI": 19, "SK": 24, "SM": 27, "ST": 25, "SV": 28,
    "TL": 23, "TN": 24, "TR": 26, "UA": 29, "VA": 22, "VG": 24, "XK": 20,
}  # fmt: skip

# Zero-width lookahead so every start position is tried, including overlapping ones.
IBAN_CANDIDATE = re.compile(r"(?<![A-Z0-9])(?=([A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,32}))")
ROUTING_NUMBER_CANDIDATE = re.compile(r"(?<![\d.])\d{9}(?!\d)")
ROUTING_NUMBER_PREFIXES = {
    *range(1, 13),  # Federal Reserve banks
    *range(21, 33),  # thrift institutions
    *range(61, 73),  # electronic transactions
    80,  # traveler's cheques
}
# US account numbers have no standard format or checksum; below 8 digits
# this matches years, form numbers, and amounts.
ACCOUNT_NUMBER_CANDIDATE = re.compile(r"(?<![\d.])\d{8,17}(?!\d)")

HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)")


@dataclass(frozen=True)
class Finding:
    """A sensitive value found at a character span within one line."""

    kind: str
    value: str
    start: int
    end: int

    def overlaps(self, other: "Finding") -> bool:
        return self.start < other.end and other.start < self.end


Detector = Callable[[str], Iterable[Finding]]


def is_valid_iban(compact: str) -> bool:
    """Check country length and ISO 13616 mod-97 checksum."""
    if len(compact) != IBAN_LENGTHS.get(compact[:2]):
        return False
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(char, 36)) for char in rearranged)) % 97 == 1


def is_valid_routing_number(digits: str) -> bool:
    """Check ABA prefix range and 3-7-1 weighted checksum."""
    if int(digits[:2]) not in ROUTING_NUMBER_PREFIXES:
        return False
    weights = (3, 7, 1) * 3
    return sum(w * int(d) for w, d in zip(weights, digits)) % 10 == 0


def find_ibans(line: str) -> Iterable[Finding]:
    for match in IBAN_CANDIDATE.finditer(line):
        compact = match.group(1).replace(" ", "")
        expected_length = IBAN_LENGTHS.get(compact[:2])
        if expected_length and is_valid_iban(compact[:expected_length]):
            end = _end_of_compact_prefix(line, match.start(), expected_length)
            yield Finding("IBAN", compact[:expected_length], match.start(), end)


def _end_of_compact_prefix(line: str, start: int, length: int) -> int:
    end = start
    while length:
        if line[end] != " ":
            length -= 1
        end += 1
    return end


def find_routing_numbers(line: str) -> Iterable[Finding]:
    for match in ROUTING_NUMBER_CANDIDATE.finditer(line):
        if is_valid_routing_number(match.group()):
            yield Finding("US routing number", match.group(), *match.span())


def find_account_numbers(line: str) -> Iterable[Finding]:
    for match in ACCOUNT_NUMBER_CANDIDATE.finditer(line):
        yield Finding("possible US account number", match.group(), *match.span())


def make_sensitive_term_finder(terms: list[str]) -> Detector:
    pattern = re.compile(
        r"\b(?:" + "|".join(map(re.escape, terms)) + r")\b", re.IGNORECASE
    )

    def find_sensitive_terms(line: str) -> Iterable[Finding]:
        for match in pattern.finditer(line):
            yield Finding("sensitive term", match.group(), *match.span())

    return find_sensitive_terms


def load_sensitive_terms(path: Path) -> list[str]:
    if not path.is_file():
        return []
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]


def build_detectors() -> list[Detector]:
    """Ordered most to least specific; earlier detectors win overlapping spans."""
    detectors: list[Detector] = [find_ibans, find_routing_numbers, find_account_numbers]
    if terms := load_sensitive_terms(SENSITIVE_TERMS_FILE):
        detectors.append(make_sensitive_term_finder(terms))
    return detectors


def scan_line(line: str, detectors: list[Detector]) -> list[Finding]:
    if ALLOW_MARKER in line:
        return []
    kept: list[Finding] = []
    for detector in detectors:
        for finding in detector(line):
            if not any(finding.overlaps(earlier) for earlier in kept):
                kept.append(finding)
    return kept


def mask(value: str) -> str:
    """Keep CI logs from re-leaking what was found."""
    if len(value) <= 6:
        return "*" * len(value)
    return value[:2] + "*" * (len(value) - 4) + value[-2:]


def format_report(location: str, finding: Finding) -> str:
    return f"{location}: {finding.kind} {mask(finding.value)}"


def scan_files(paths: list[str], detectors: list[Detector]) -> Iterator[str]:
    for path in paths:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for finding in scan_line(line, detectors):
                yield format_report(f"{path}:{line_number}", finding)


def scan_history(pathspecs: list[str], detectors: list[Detector]) -> Iterator[str]:
    """Scan lines added in every commit on every branch."""
    log = subprocess.run(
        ["git", "log", "--all", "-p", "--no-color", "--format=commit %h", "--", "."]
        + pathspecs,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    ).stdout
    commit = path = ""
    line_number = 0
    for line in log.splitlines():
        if line.startswith("commit "):
            commit = line.removeprefix("commit ")
        elif line.startswith("+++ "):
            path = line.removeprefix("+++ ").removeprefix("b/")
        elif hunk := HUNK_HEADER.match(line):
            line_number = int(hunk.group(1))
        elif line.startswith("+"):
            for finding in scan_line(line[1:], detectors):
                yield format_report(f"{commit}:{path}:{line_number}", finding)
            line_number += 1
        elif line.startswith(" "):
            line_number += 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", help="files (git pathspecs with --history)")
    parser.add_argument("--history", action="store_true", help="scan all git history")
    args = parser.parse_args(argv)

    detectors = build_detectors()
    scan = scan_history if args.history else scan_files
    reports = list(scan(args.paths, detectors))
    for report in reports:
        print(report)
    if reports:
        print(f"\nAppend '{ALLOW_MARKER}' to a line to accept a false positive.")
    return 1 if reports else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
