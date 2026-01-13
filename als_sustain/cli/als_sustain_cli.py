"""Command line front-end for the package.
Usage examples:
  # feature-based quick mode (features CSV paths)
  python -m als_sustain.cli.als_sustain_cli run \
    --model preselected_regions_DBM_model_testwithdataready \
    --input examples/Participant_Inputs_File_features_calsnic.csv \
    --outdir /tmp/out

  # whole brain DBM map mode (DBM paths)
  python -m als_sustain.cli.als_sustain_cli run \
    --model preselected_regions_DBM_model_testwithDBMmaps \
    --input examples/Participant_Inputs_File_features_calsnic_dbm.csv \
    --outdir /tmp/out

  # full pipeline mode (T1 paths)
  python -m als_sustain.cli.als_sustain_cli run --model preselected_regions_DBM_model --input Participant_Inputs_File_t1.csv --outdir /tmp/out --singularity-bind /data

When built into the container, provide an entrypoint that maps to the main() below.
"""
import argparse
from als_sustain.pipeline.run_pipeline import run_batch, load_descriptor
from als_sustain.utils.io import make_json_safe
from pathlib import Path
import yaml
from typing import Dict
import json
import pandas as pd



def load_resources(root_dir: Path) -> Dict:
    """Load container-internal resources configuration."""
    cfg_path = root_dir / "config" / "resources.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Resources file not found: {cfg_path}")
   
    with cfg_path.open() as f:
        return yaml.safe_load(f)

def main():
    # Get the project root directory (assuming this file is in als_sustain_inference/als_sustain/cli/)
    root_dir = Path(__file__).resolve().parent.parent.parent

    parser = argparse.ArgumentParser(description='ALS SuStaIn multi-model CLI')

    subparsers = parser.add_subparsers(dest='command')

    run_parser = subparsers.add_parser('run', help='Run model on batch CSV')
    run_parser.add_argument('--model', 
                            type=str,
                            required=True, 
                            help='Model ID (descriptor filename without .yaml)')
    run_parser.add_argument('--input_filepath',
                            type=Path, 
                            required=True, 
                            help='Path to Participant_Inputs.csv (ID, Visit, Path)')
    run_parser.add_argument('--input_type', 
                            type=str, 
                            required=True,
                            help='Type of data pointed to by the "Path" column')
    run_parser.add_argument('--outdir', 
                            type=Path, 
                            default=Path.cwd(), 
                            help='Directory to save results (default: current directory)')
    run_parser.add_argument('--show_debug_outputs', 
                            type=bool, 
                            default=False, 
                            help='If True, save intermediate debug outputs (default: False)') 
    
    args = parser.parse_args()

    if args.command == 'run':
        outdir = args.outdir.resolve()
        outdir.mkdir(parents=True, exist_ok=True)

        if args.show_debug_outputs:
            debug_dir = outdir / "debug_outputs"
            debug_dir.mkdir(parents=True, exist_ok=True)
        else:
            debug_dir = None

        resources = load_resources(root_dir=root_dir)

        if not isinstance(resources, dict):
          raise RuntimeError("Failed to load resources config")
       
        # Validate --input_type against the model descriptor
        desc = load_descriptor(model_id=args.model, root_dir=root_dir)
        accepted = desc.get("processing_input_accepted", [])
        if args.input_type not in accepted:
            raise ValueError(f"input_type {args.input_type} not accepted, expected one of: {accepted}")

        results = run_batch(
            input_csv=args.input_filepath,
            input_type=args.input_type,
            model_id=args.model,
            outdir=outdir,
            resources=resources,
            root_dir=root_dir,
            debug_dir=debug_dir,
        )

        print('--- Summary ---')
        print(yaml.safe_dump(results, sort_keys=False))
        results_df = pd.json_normalize(results, sep=".")
        results_df.columns = [
            c.replace("prediction.", "") for c in results_df.columns
        ]
        results_df.to_csv(outdir / "results_summary.csv", index=False)

if __name__ == '__main__':
    main()
