# 1) Base Container
ARG BASE_CONTAINER=ucsdets/datahub-base-notebook:2022.3-stable
FROM $BASE_CONTAINER
LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) Change to root to install packages
USER root
RUN apt update
RUN apt-get -y install aria2 nmap traceroute

# 3) Install packages using notebook user
USER jovyan
RUN pip install --no-cache-dir torch pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
RUN pip install --no-cache-dir pandas torch-geometric pyTigerGraph numpy scikit-learn matplotlib xgboost networkx

# 4) Disable running jupyter notebook at launch
CMD ["/bin/bash"]