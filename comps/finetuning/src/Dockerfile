# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Use the same python version with ray
FROM python:3.10.14

ARG HF_TOKEN

ENV HF_TOKEN=$HF_TOKEN

RUN useradd -m -s /bin/bash user && \
    mkdir -p /home/user && \
    chown -R user /home/user/

COPY comps /home/user/comps

RUN chown -R user /home/user/comps/finetuning

ENV PATH=$PATH:/home/user/.local/bin

RUN python -m pip install --no-cache-dir --upgrade pip setuptools && \
    python -m pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    python -m pip install --no-cache-dir intel-extension-for-pytorch && \
    python -m pip install --no-cache-dir oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/ && \
    python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r /home/user/comps/finetuning/src/requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home/user

USER user

WORKDIR /home/user/comps/finetuning/src

RUN echo PKGPATH=$(python3 -c "import pkg_resources; print(pkg_resources.get_distribution('oneccl-bind-pt').location)") >> run.sh && \
    echo 'export LD_LIBRARY_PATH=$PKGPATH/oneccl_bindings_for_pytorch/opt/mpi/lib/:$LD_LIBRARY_PATH' >> run.sh && \
    echo 'source $PKGPATH/oneccl_bindings_for_pytorch/env/setvars.sh' >> run.sh && \
    echo ray start --head --dashboard-host=0.0.0.0 >> run.sh && \
    echo export RAY_ADDRESS=http://localhost:8265 >> run.sh && \
    echo python opea_finetuning_microservice.py >> run.sh

CMD ["bash", "run.sh"]
