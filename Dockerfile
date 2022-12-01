FROM python:3.10.1-slim

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/streamlit/streamlit-example.git .

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "server.py", "--server.port=8501", "--server.address=0.0.0.0"]


FROM condaforge/miniforge3
LABEL maintainer: "FSL development team <https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/>"
ENV PATH="/opt/conda/bin:${PATH}"

# store the FSL public conda channel
ENV FSL_CONDA_CHANNEL="https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public"
# entrypoint activates conda environment and fsl when you `docker run <command>`
COPY /entrypoint /entrypoint
# make entrypoint executable
RUN chmod +x /entrypoint
# install tini into base conda environment
RUN /opt/conda/bin/conda install -n base -c conda-forge tini
# as a demonstration, install ONLY FSL's bet (brain extraction) tool. This is an example of a minimal, yet usable container without the rest of FSL being installed
# to see all packages available use a browser to navigate to: https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/public/
# note the channel priority. The FSL conda channel must be first, then condaforge
RUN /opt/conda/bin/conda install -n base -c $FSL_CONDA_CHANNEL fsl-bet2 -c conda-forge
# set FSLDIR so FSL tools can use it, in this minimal case, the FSLDIR will be the root conda directory
ENV FSLDIR="/opt/conda"

ENTRYPOINT [ "/opt/conda/bin/tini", "--", "/entrypoint" ]
