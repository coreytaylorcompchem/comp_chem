#!/usr/bin/env python3

import pdfplumber
import re
import pandas as pd
from pathlib import Path


#############################################
# TABLE CATEGORY → NAME SUFFIX RULES
#############################################

ACID_CLASSES = {
    "dicarboxylic acids": "acid",
    "carboxylic acids": "acid",
    "unsaturated acids": "acid",
    "alicyclic dicarboxylic acids": "acid",
    "dicarboxylic acids, unsaturated": "acid",
    "aliphatic acids": "acid",
    "aromatic acids": "acid",
    "amino acids": "acid",
    "thiols": None,
    "phosphates": None,
    "phosphonates": None,
    "aliphatic amines": None,
    "aromatic amines": None,
}


def detect_table_title(line):
    """Return normalized table class if line looks like a heading."""
    clean = line.lower().strip()
    for k in ACID_CLASSES.keys():
        if k in clean:
            return k
    return None


def repair_name(name, current_class):
    """Append 'acid' or other suffix if the table header indicates missing context."""
    if not current_class or current_class not in ACID_CLASSES:
        return name

    suffix = ACID_CLASSES[current_class]
    if suffix is None:
        return name

    # name ends with typical full chemical names? If so don't modify.
    if re.search(r"(acid|ate|one|ol|ene|ide|ium|oxide|amide|amine|thiol)$",
                 name.lower()):
        return name

    # e.g. "Oxalic" → "Oxalic acid"
    return f"{name} {suffix}"


#############################################
# LINE PARSER
#############################################

PKA_RE = re.compile(r"-?\d+\.\d+\*?")
NAME_RE = re.compile(r"^[A-Za-z0-9\-\(\)\/\+\=\[\]\,\.]+$")

def parse_line(line, current_class):
    """
    Extract {name, pkas:[], ref}.
    Return None if line doesn't contain pKa-like data.
    """
    original = line
    line = line.replace("–", "-").replace("—", "-")
    line = line.replace(",", ", ")
    line = re.sub(r"\s+", " ", line).strip()

    # Extract pKa values
    pka_vals = PKA_RE.findall(line)

    if not pka_vals:
        return None

    # Clean pKa values
    pkas = []
    for v in pka_vals:
        num = v.replace("*", "")
        pkas.append(float(num))

    # Remove pKas from the line
    line_wo_pka = PKA_RE.sub("", line).strip()

    # Extract ref (last number in string)
    ref_match = re.search(r"(\d+)$", line_wo_pka)
    ref = int(ref_match.group(1)) if ref_match else None

    # Remove trailing ref
    if ref:
        line_wo_pka = re.sub(rf"{ref}\s*$", "", line_wo_pka).strip()

    # Remaining text is name
    name = line_wo_pka.strip()
    name = name.strip("-").strip()

    # Fix truncated names based on table header
    name = repair_name(name, current_class)

    return {
        "name": name,
        "pkas": pkas,
        "ref": ref
    }


#############################################
# COLUMN DETECTION
#############################################

def detect_columns(page):
    """Heuristic: determine whether a page has two columns."""
    chars = page.chars
    if not chars:
        return 1

    xs = [c["x0"] for c in chars]
    mid = page.width / 2

    left = sum(1 for x in xs if x < mid)
    right = sum(1 for x in xs if x >= mid)

    # 2-column if both sides have substantial text
    if left > 0.55 * (left + right) and right > 0.15 * (left + right):
        return 2
    return 1


def split_two_columns(page):
    width = page.width
    height = page.height
    mid = width / 2

    left_bbox  = (0, 0, mid, height)
    right_bbox = (mid, 0, width, height)

    left_text = page.crop(left_bbox).extract_text()
    right_text = page.crop(right_bbox).extract_text()

    return [left_text, right_text]


#############################################
# MAIN EXTRACTION
#############################################

def extract_pka_data(pdf_path):
    rows = []
    pdf_path = Path(pdf_path)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            col_count = detect_columns(page)

            if col_count == 1:
                col_texts = [page.extract_text()]
            else:
                col_texts = split_two_columns(page)

            current_class = None

            for col_text in col_texts:
                if not col_text:
                    continue

                for line in col_text.split("\n"):
                    # Update table class if line is a heading
                    new_class = detect_table_title(line)
                    if new_class:
                        current_class = new_class
                        continue

                    entry = parse_line(line, current_class)
                    if entry:
                        entry["page"] = page_num
                        rows.append(entry)

    return rows


#############################################
# REFORMAT INTO LONG FORMAT
#############################################

def rows_to_long_df(rows):
    df = pd.DataFrame(rows)

    # site numbers for multi-pKa species
    df["site"] = df["pkas"].apply(lambda lst: list(range(1, len(lst)+1)))

    long_df = df.explode(["pkas", "site"]).rename(columns={"pkas": "pKa"})

    # Clean columns
    long_df["name"] = long_df["name"].str.strip()
    long_df = long_df[["name", "site", "pKa", "ref", "page"]]

    return long_df


#############################################
# RUN SCRIPT
#############################################

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract tidy pKa dataset from ACS PDF")
    parser.add_argument("pdf", help="Path to ACS pKa PDF")
    parser.add_argument("-o", "--output", default="pka_clean_long.csv",
                        help="Output CSV filename")

    args = parser.parse_args()

    print("Extracting pKa data...")
    rows = extract_pka_data(args.pdf)

    print("Converting to long format...")
    df = rows_to_long_df(rows)

    print(f"Saving to {args.output}")
    df.to_csv(args.output, index=False)

    print("Done.")
