FROM continuumio/miniconda3:latest

# --- 1. Install System Dependencies ---
RUN apt-get update && apt-get install -y \
    perl \
    bash wget curl git ca-certificates \
    build-essential g++ \
    libgl1 libglu1-mesa libx11-6 \
    bc imagemagick ghostscript \
    libfreetype6-dev libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# --- 2. Install MINC Toolkit (Linux Version) ---
# We download the Ubuntu 18.04/20.04 64-bit Linux version 
# which is compatible with the Debian-based miniconda image.
RUN wget --no-check-certificate https://packages.bic.mni.mcgill.ca/minc-toolkit/Debian/minc-toolkit-1.9.18-20200813-Ubuntu_18.04-x86_64.deb -O /tmp/minc.deb && \
    apt-get update && \
    apt-get install -y /tmp/minc.deb && \
    rm /tmp/minc.deb

# Verify the actual path where the .deb installs itself. 
# Usually it is /opt/minc/1.9.18 or /opt/minc-itk4/1.9.18
ENV MINC_TOOLKIT=/opt/minc/1.9.18
ENV ANTSPATH=$MINC_TOOLKIT/bin
ENV PATH=$MINC_TOOLKIT/bin:$PATH
ENV MNI_DATAPATH=$MINC_TOOLKIT/share
ENV PERL5LIB=$MINC_TOOLKIT/perl:$MINC_TOOLKIT/dev:/usr/share/perl5
ENV MINC_FORCE_V2=1
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV LD_LIBRARY_PATH=$MINC_TOOLKIT/lib:$LD_LIBRARY_PATH

# --- 3. Setup Python Environments ---
RUN conda create -n pelican_env python=3.9 -y && \
    /opt/conda/envs/pelican_env/bin/pip install \
    SimpleITK \
    scikit-learn==1.0.2 \
    joblib==1.1.0 \
    numpy==1.22.4


RUN conda create -n main_env python=3.9 -y && \
    /opt/conda/bin/conda install -n main_env -y \
    -c conda-forge \
    numpy=1.26.4 \
    pandas=2.2.2 \
    scikit-learn=1.5.2 \
    nibabel=5.2.1 \
    seaborn \
    scikit-survival \
    statsmodels \
    joblib \
    pyyaml \
    tqdm && \ 
    /opt/conda/envs/main_env/bin/pip install \
    git+https://github.com/ucl-pond/pySuStaIn \
    requests


# --- 4. Project Files & Permissions ---
WORKDIR /app
COPY . /app

# Install the package 
RUN /opt/conda/envs/main_env/bin/pip install --no-deps .

# Ensure Pelican binaries are executable
RUN chmod -R +x /app/als_prognosis/backends/pelican/

# Create the neutral data directory for pelican and for libraries (Matplotlib, Fontconfig, Conda)
# This prevents "Permission Denied" when running as a non-root user
RUN mkdir -p /data/pelican \
    /tmp/matplotlib \
    /tmp/fontconfig \
    /tmp/.cache \
    /opt/conda/pkgs && \
    chmod -R 777 /data /tmp /opt/conda/pkgs /root

# Put the environment variables into the image
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV FONTCONFIG_PATH=/tmp/fontconfig
# For Fontconfig and other Linux tools:
ENV XDG_CACHE_HOME=/tmp/.cache

ENV PATH="/opt/conda/envs/main_env/bin:$PATH"

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "main_env", "python", "-m", "als_prognosis.cli.als_prognosis_cli", "run"]

