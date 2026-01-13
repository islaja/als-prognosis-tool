import subprocess
import gzip
import shutil
from pathlib import Path

def minc2nii(input_path:str, output_path:str): #-> str:
    """Convert a MINC file to NIfTI format using an external tool (e.g., mnc2nii).

    Parameters
    ----------
    input_path: path to input MINC file
    output_path: path to output NIfTI file
    output_format: format flag for mnc2nii (default '-f' for .nii)
    """

    # Convert to .nii first
    output_tmp = output_path.replace('.gz', '')
  
    subprocess.run(["mnc2nii", "-float", "-nii", input_path, output_tmp], check=True)

    if output_path.endswith('.gz'):
        # Compress to .gz  
        with open(output_tmp, 'rb') as f_in:
            with gzip.open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Remove the uncompressed temp file
        Path(output_tmp).unlink()
    
    """if not input_path.endswith('.mnc'):
        raise ValueError("Input file must be a MINC file with .mnc extension")  
    if not output_path_nii.endswith(('.nii', '.nii.gz')):
        raise ValueError("Output file must be a NIfTI file with .nii or .nii.gz extension") 
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input MINC file {input_path} does not exist") 
    
    cmd = ["mnc2nii", output_format, input_path, output_path]
    subprocess.check_call(cmd)   """   
    #return output_path