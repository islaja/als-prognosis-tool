"""Command line front-end for the package.
Usage examples:
  # feature-based quick mode (features CSV paths)
  python -m als_sustain.cli.als_sustain_cli run \
    --model preselected_regions_DBM_model_testwithdataready \
    --input examples/Participant_Inputs_File_features_calsnic.csv \
    --workdir /tmp/out

  # whole brain DBM map mode (DBM paths)
  python -m als_sustain.cli.als_sustain_cli run \
    --model preselected_regions_DBM_model_testwithDBMmaps \
    --input examples/Participant_Inputs_File_features_calsnic_dbm.csv \
    --workdir /tmp/out

  # full pipeline mode (T1 paths)
  python -m als_sustain.cli.als_sustain_cli run --model preselected_regions_DBM_model --input Participant_Inputs_File_t1.csv --workdir /tmp/out --singularity-bind /data

When built into the container, provide an entrypoint that maps to the main() below.
"""
import argparse
from als_sustain.pipeline.run_pipeline import run_batch
from als_sustain.utils.io import make_json_safe
from pathlib import Path
import yaml
from typing import Dict
import json
import pandas as pd


def load_resources(base_dir: Path) -> Dict:
    """Load container-internal resources configuration."""
    cfg_path = base_dir / "config" / "resources.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Resources file not found: {cfg_path}")
   
    with cfg_path.open() as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='ALS SuStaIn multi-model CLI')

    subparsers = parser.add_subparsers(dest='command')

    run_parser = subparsers.add_parser('run', help='Run model on inputs CSV')
    run_parser.add_argument('--model', required=True, help='Model ID (descriptor filename without .yaml)')
    run_parser.add_argument('--input', required=True, help='Participant_Inputs_File.csv')
    run_parser.add_argument('--workdir', default='./workdir', help='Output working directory')
    run_parser.add_argument('--base-dir', default='.', help='Base repo dir (where models/config live)')

    args = parser.parse_args()

    if args.command == 'run':
        base_dir = Path(args.base_dir).resolve()
        workdir = (base_dir / Path(args.workdir)).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        resources = load_resources(base_dir)

        if not isinstance(resources, dict):
          raise RuntimeError("Failed to load resources config")
       
        results = run_batch(
            input_csv=args.input, 
            model_id=args.model, 
            workdir=workdir, 
            resources=resources,
            base_dir=base_dir,
            )
        print('--- Summary ---')
        print(yaml.safe_dump(results, sort_keys=False))
        results_df = pd.json_normalize(results, sep=".")
        results_df.columns = [
            c.replace("prediction.", "") for c in results_df.columns
        ]
        results_df.to_csv(workdir / "results_summary.csv", index=False)

if __name__ == '__main__':
    main()
