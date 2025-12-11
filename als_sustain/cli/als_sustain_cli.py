"""Command line front-end for the package.
Usage examples:
  # feature-based quick mode (features CSV paths)
  python -m als_sustain.cli.als_sustain_cli run --model preselected_regions_DBM_model_testwithdataready --input Participant_Inputs_File_features_calsnic.csv --workdir /tmp/out

  # full pipeline mode (T1 paths)
  python -m als_sustain.cli.als_sustain_cli run --model preselected_regions_DBM_model --input Participant_Inputs_File_t1.csv --workdir /tmp/out --singularity-bind /data

When built into the container, provide an entrypoint that maps to the main() below.
"""
import argparse
import os
from als_sustain.pipeline.run_pipeline import run_batch
from als_sustain.utils.io import make_json_safe


def main():
    parser = argparse.ArgumentParser(description='ALS SuStaIn multi-model CLI')

    subparsers = parser.add_subparsers(dest='command')

    run_parser = subparsers.add_parser('run', help='Run model on inputs CSV')
    run_parser.add_argument('--model', required=True, help='Model ID (descriptor filename without .json)')
    run_parser.add_argument('--input', required=True, help='Participant_Inputs_File.csv')
    run_parser.add_argument('--workdir', default='./workdir', help='Output working directory')
    run_parser.add_argument('--base-dir', default='.', help='Base repo dir (where models/config live)')

    args = parser.parse_args()

    if args.command == 'run':
        os.makedirs(args.workdir, exist_ok=True)
        results = run_batch(args.input, args.model, args.workdir, base_dir=args.base_dir)
        # make JSON-safe before printing
        results_safe = make_json_safe(results)
        import json
        print(json.dumps(results_safe, indent=2))

if __name__ == '__main__':
    main()
