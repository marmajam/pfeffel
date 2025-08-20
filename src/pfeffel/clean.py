import argparse
import json
import os
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd
import yaml

from src.clean_data import load_clean_data


# Resolve important project paths from this file's location
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
CONFIG_FPATH = os.path.join(PROJECT_ROOT, "config.yaml")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def create_clean_data(
    bikefolder: str,
    output_dir: str,
    num_files: Optional[int] = None,
    datapaths: Optional[List[str]] = None,
    write_parquet: bool = False,
    timestamp_output: bool = True,
) -> Tuple[str, Optional[str], str]:
    """
    Use load_clean_data to build the cleaned dataset and persist it to disk.

    Returns tuple of (pickle_fpath, parquet_fpath_or_None, station_map_json_fpath)
    """
    print(f"Loading raw trips from: {bikefolder}")
    df, station_id_to_names = load_clean_data(
        bikefolder=bikefolder, num_files=num_files, datapaths=datapaths
    )

    _ensure_dir(output_dir)

    ts = _timestamp() if timestamp_output else None
    base_pickle_name = (
        f"cleaned_data_{ts}.pickle" if ts else "cleaned_data.pickle"
    )
    base_parquet_name = (
        f"cleaned_data_{ts}.parquet" if ts else "cleaned_data.parquet"
    )
    base_station_map_name = (
        f"station_id_to_names_{ts}.json" if ts else "station_id_to_names.json"
    )
    base_station_map_pickle_name = (
        f"station_id_to_names_{ts}.pickle" if ts else "station_id_to_names.pickle"
    )
    base_station_names_pickle_name = (
        f"station_names_{ts}.pickle" if ts else "station_names.pickle"
    )

    pickle_fpath = os.path.join(output_dir, base_pickle_name)
    parquet_fpath = os.path.join(output_dir, base_parquet_name)
    station_map_fpath = os.path.join(output_dir, base_station_map_name)
    station_map_pickle_fpath = os.path.join(output_dir, base_station_map_pickle_name)
    # Raw mapping with sets (for compatibility with older notebooks/code)
    station_names_pickle_fpath = os.path.join(output_dir, base_station_names_pickle_name)

    print(f"Writing pickle: {pickle_fpath}")
    df.to_pickle(pickle_fpath)

    if write_parquet:
        print(f"Writing parquet: {parquet_fpath}")
        try:
            # Ensure index is preserved; avoid nullable dtypes issues by letting pandas handle
            df.to_parquet(parquet_fpath, index=True)
        except Exception as e:
            print(
                "Failed to write parquet (missing engine like 'pyarrow' or 'fastparquet'?).\n"
                f"Skipping parquet. Error: {e}"
            )
            parquet_fpath = None
    else:
        parquet_fpath = None

    # Convert sets to sorted lists for JSON serialization
    serializable_station_map = {
        int(k): sorted(list(v)) for k, v in station_id_to_names.items()
    }
    print(f"Writing station ID->names map: {station_map_fpath}")
    with open(station_map_fpath, "w", encoding="utf-8") as f:
        json.dump(serializable_station_map, f, ensure_ascii=False, indent=2)
    # Also write as pickle for easy loading in notebooks
    print(f"Writing station ID->names pickle: {station_map_pickle_fpath}")
    pd.to_pickle(serializable_station_map, station_map_pickle_fpath)
    # Write raw set-valued mapping as a separate pickle to retain original structure
    print(f"Writing raw station names pickle (set-valued): {station_names_pickle_fpath}")
    pd.to_pickle(station_id_to_names, station_names_pickle_fpath)

    # Also maintain stable "latest" copies for convenience
    latest_pickle = os.path.join(output_dir, "cleaned_data_latest.pickle")
    print(f"Writing latest pickle: {latest_pickle}")
    df.to_pickle(latest_pickle)

    if write_parquet and parquet_fpath is not None:
        latest_parquet = os.path.join(output_dir, "cleaned_data_latest.parquet")
        print(f"Writing latest parquet: {latest_parquet}")
        try:
            df.to_parquet(latest_parquet, index=True)
        except Exception as e:
            print(
                "Failed to write latest parquet (missing engine?). Skipping.\n"
                f"Error: {e}"
            )

    latest_station_map = os.path.join(output_dir, "station_id_to_names.json")
    latest_station_map_pickle = os.path.join(output_dir, "station_id_to_names_latest.pickle")
    latest_station_names_pickle = os.path.join(output_dir, "station_names_latest.pickle")
    print(f"Writing latest station map: {latest_station_map}")
    with open(latest_station_map, "w", encoding="utf-8") as f:
        json.dump(serializable_station_map, f, ensure_ascii=False, indent=2)
    print(f"Writing latest station map pickle: {latest_station_map_pickle}")
    pd.to_pickle(serializable_station_map, latest_station_map_pickle)
    print(f"Writing latest raw station names pickle: {latest_station_names_pickle}")
    pd.to_pickle(station_id_to_names, latest_station_names_pickle)

    return pickle_fpath, parquet_fpath, station_map_fpath


def main() -> None:
    parser = argparse.ArgumentParser(description="Create cleaned Boris Bikes trip dataset")
    parser.add_argument(
        "--csv-dir",
        dest="csv_dir",
        type=str,
        default=None,
        help="Directory containing raw trip CSVs (defaults to config.data.relative_paths.csvs_dir)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        type=str,
        default=None,
        help="Directory to write cleaned outputs (defaults to config.data.root_dir)",
    )
    parser.add_argument(
        "--num-files",
        dest="num_files",
        type=int,
        default=None,
        help="Optional limit on the number of CSV files to process (for quick runs)",
    )
    parser.add_argument(
        "--no-parquet",
        dest="write_parquet",
        action="store_false",
        help="Disable writing parquet output",
    )
    parser.add_argument(
        "--no-timestamp",
        dest="timestamp_output",
        action="store_false",
        help="Disable timestamped filenames (only write '*_latest.*')",
    )
    parser.add_argument(
        "--only-stations-pickle",
        dest="only_stations_pickle",
        action="store_true",
        help="Only (re)create station_names pickle files; do not write cleaned data or other files",
    )

    args = parser.parse_args()

    # Load configuration for default paths
    with open(CONFIG_FPATH, "r") as f:
        config = yaml.safe_load(f)

    data_root_dir = os.path.join(PROJECT_ROOT, config["data"]["root_dir"])  # e.g., <repo>/data
    default_csv_dir = os.path.join(
        data_root_dir, config["data"]["relative_paths"]["csvs_dir"]
    )

    csv_dir = args.csv_dir or default_csv_dir
    output_dir = args.output_dir or data_root_dir

    print(f"Using CSV dir: {csv_dir}")
    print(f"Using output dir: {output_dir}")

    if args.only_stations_pickle:
        # Build only the station_names pickles
        print("Building only station names pickles (no cleaned data will be written)...")
        _, station_id_to_names = load_clean_data(
            bikefolder=csv_dir, num_files=args.num_files, datapaths=None
        )

        ts = _timestamp() if args.timestamp_output else None
        base_station_names_pickle_name = (
            f"station_names_{ts}.pickle" if ts else "station_names.pickle"
        )
        station_names_pickle_fpath = os.path.join(
            output_dir, base_station_names_pickle_name
        )
        latest_station_names_pickle = os.path.join(
            output_dir, "station_names_latest.pickle"
        )

        print(
            f"Writing raw station names pickle (set-valued): {station_names_pickle_fpath}"
        )
        pd.to_pickle(station_id_to_names, station_names_pickle_fpath)
        print(
            f"Writing latest raw station names pickle: {latest_station_names_pickle}"
        )
        pd.to_pickle(station_id_to_names, latest_station_names_pickle)
        return

    create_clean_data(
        bikefolder=csv_dir,
        output_dir=output_dir,
        num_files=args.num_files,
        datapaths=None,
        write_parquet=bool(args.write_parquet),
        timestamp_output=bool(args.timestamp_output),
    )


if __name__ == "__main__":
    main()


