FROM ucsdets/scipy-ml-notebook:2023.1-stable

USER jovyan

RUN pip install --upgrade pip \
    && pip install \
        pyTigerGraph \
        enum34==1.1.8 \
        xgboost \ 
        torch-geometric

CMD ["/bin/bash"]