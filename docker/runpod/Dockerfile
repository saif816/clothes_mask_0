FROM mambaorg/micromamba:2.3.0 AS micromamba

FROM nvcr.io/nvidia/tensorrt:25.06-py3

COPY --from=micromamba /bin/micromamba /bin/micromamba
ENV MAMBA_ROOT_PREFIX=/opt/conda  \
    MAMBA_EXE=/bin/micromamba     \
    PATH=/opt/conda/bin:$PATH

RUN micromamba install -y -n base -c conda-forge \
        python=3.10  pip  setuptools wheel && \
    micromamba clean --all --yes

SHELL ["/bin/bash", "-l", "-c"]
RUN echo "micromamba activate base" >> ~/.bashrc

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /app

COPY Models/ /app/

COPY src/ /app/

COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python3", "-u", "handlerbatch.py"]