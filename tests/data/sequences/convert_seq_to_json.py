#!/usr/bin/env python3
"""
Script to convert MAD-X sequence files (.seq) to xsuite JSON format.

This script reads all .seq files in the current directory and generates
corresponding .json files using xsuite's load_madx_lattice function.
"""

import logging
from pathlib import Path

import xtrack as xt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_seq_to_json(seq_file: Path) -> None:
    """Convert a single .seq file to .json format."""
    json_file = seq_file.with_suffix('.json')

    if json_file.exists():
        logger.info(f"JSON file already exists: {json_file}")
        return

    logger.info(f"Converting {seq_file} to {json_file}")

    try:
        # Load MAD-X lattice from sequence file
        env = xt.load(file=seq_file)

        # Save to JSON
        env.to_json(json_file)

        logger.info(f"Successfully converted {seq_file} to {json_file}")

    except Exception as e:
        logger.error(f"Failed to convert {seq_file}: {e}")
        raise

def main() -> None:
    """Convert all .seq files in the current directory to JSON."""
    current_dir = Path(__file__).parent

    seq_files = list(current_dir.glob("*.seq"))
    if not seq_files:
        logger.warning("No .seq files found in the current directory")
        return

    logger.info(f"Found {len(seq_files)} .seq files to convert")

    for seq_file in seq_files:
        convert_seq_to_json(seq_file)

    logger.info("Conversion complete")

if __name__ == "__main__":
    main()
