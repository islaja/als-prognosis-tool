"""
ALS SuStaIn CLI module.

Provides a command-line interface for running SuStaIn models on batch CSV files
or full pipeline inputs (features, DBM maps, or T1 images).

Usage examples:
  # dbm-wscores-based 
  python -m als_prognosis.cli.als_prognosis_cli run \    
    --model CALSNIC_sustain_14_reg_dbm_wscore \
    --input_filepath examples/Participant_Inputs_File_features_calsnic.csv \
    --input_type regional_dbm_wscores \
    --outdir tmp/from_dbm_wscores \

  # whole brain DBM map mode (DBM paths)
  python -m als_prognosis.cli.als_prognosis_cli run \
    --model CALSNIC_sustain_14_reg_dbm_wscore \
    --input_filepath examples/Participant_Inputs_File_features_calsnic_dbm_maps.csv \
    --input_type dbm_maps \
    --outdir tmp/from_dbm_maps \
    --show_debug_outputs True

  # full pipeline mode (T1 paths)
  python python -m als_prognosis.cli.als_prognosis_cli run \
    --model CALSNIC_sustain_14_reg_dbm_wscore \
    --input_filepath examples/Participant_Inputs_File_features_calsnic_t1w_maps.csv \
    --input_type t1w_maps \
    --outdir tmp/from_t1w_maps \
    --show_debug_outputs True

When built into the container, provide an entrypoint that maps to the main() below.
"""

import argparse
from pathlib import Path
from typing import Dict

import yaml
import pandas as pd

from als_prognosis.pipeline.run_pipeline import run_batch, load_descriptor

import logging
import sys

def setup_initial_logging():
    """ Console only """
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    
    # Clean terminal output
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter('%(message)s'))
    root.addHandler(console)
    
    # Mute noisy libraries
    logging.getLogger('nibabel').setLevel(logging.ERROR)


def add_file_logging(outdir: Path):
    """ Attach file logging once outdir is known """
    log_path = outdir / "pipeline.log"
    
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    
    logging.getLogger().addHandler(file_handler)
    return log_path


def load_resources(root_dir: Path) -> Dict:
    """
    Load container-internal resources configuration.

    Args:
        root_dir (Path): Project root directory where config/resources.yaml resides.

    Returns:
        Dict: Parsed YAML resources configuration.

    Raises:
        FileNotFoundError: If resources.yaml is missing.
    """
    cfg_path = root_dir / "config" / "resources.yaml"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Resources file not found: {cfg_path}")
   
    with cfg_path.open() as f:
        return yaml.safe_load(f)

def main():
    """
    Entry point for ALS SuStaIn CLI.

    Parses command-line arguments, validates inputs against the model descriptor,
    runs the selected model pipeline, and outputs results (CSV).

    Supports:
        - roi-based mode
        - DBM map mode
        - Full T1 pipeline mode
    """
    # Start console logging
    setup_initial_logging()
    logger = logging.getLogger(__name__)

    # Determine project root (assumes this CLI module is at als_prognosis/cli/)
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

        # Enable file logging
        log_file = add_file_logging(outdir)
        display_path = f"{log_file.parent.name}/{log_file.name}"  
        logger.info(f"\n📄 Log is being saved to: {display_path}")

        # Create debug output directory if requested
        if args.show_debug_outputs:
            debug_dir = outdir / "debug_outputs"
            debug_dir.mkdir(parents=True, exist_ok=True)
        else:
            debug_dir = None

        # Load resources configuration
        resources = load_resources(root_dir=root_dir)

        if not isinstance(resources, dict):
          raise RuntimeError("Failed to load resources config")
       
        # Load model descriptor and validate input type
        desc = load_descriptor(model_id=args.model, root_dir=root_dir)
        accepted = desc.get("processing_input_accepted", [])
        if args.input_type not in accepted:
            raise ValueError(f"input_type {args.input_type} not accepted, expected one of: {accepted}")

        # Run the model batch
        results = run_batch(
            input_csv=args.input_filepath,
            input_type=args.input_type,
            model_id=args.model,
            outdir=outdir,
            resources=resources,
            root_dir=root_dir,
            debug_dir=debug_dir,
        )

        # Print YAML summary
        #print('--- Summary ---')
        #print(yaml.safe_dump(results, sort_keys=False))
        
        # Flatten nested prediction keys and save CSV summary
        results_df = pd.json_normalize(results, sep=".")
        results_df.columns = [
            c.replace("sustain_inference.", "") for c in results_df.columns
        ]
        summary_path = outdir / "results_summary.csv"
        display_path = f"{summary_path.parent.name}/{summary_path.name}"
        logger.info(f"\n📊 Summary of inference and prediction found at: {display_path}\n")
        results_df.to_csv(summary_path, index=False)

if __name__ == '__main__':
    main()
