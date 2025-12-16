def minc2nii(input_path, output_path, output_format='-f'):
    """Convert a MINC file to NIfTI format using an external tool (e.g., mnc2nii).

    Parameters
    ----------
    input_path: path to input MINC file
    output_path: path to output NIfTI file
    output_format: format flag for mnc2nii (default '-f' for .nii)
    """
    import os
    import subprocess
    
    if not input_path.endswith('.mnc'):
        raise ValueError("Input file must be a MINC file with .mnc extension")  
    if not output_path.endswith(('.nii', '.nii.gz')):
        raise ValueError("Output file must be a NIfTI file with .nii or .nii.gz extension") 
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input MINC file {input_path} does not exist") 
    
    cmd = ["mnc2nii", output_format, input_path, output_path]
    subprocess.check_call(cmd)      
    return output_path