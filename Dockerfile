FROM python:3.10-slim

EXPOSE 8501

WORKDIR /DS1016-Project

#RUN apt-get update && apt-get install -y \
#    build-essential \
#    software-properties-common \
#    git \
#    && rm -rf /var/lib/apt/lists/*

#RUN git clone https://github.com/Shaurynn/DS1016-Project.git .
COPY . .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN python fslinstaller.py -d /usr/local/fsl
RUN export FSLDIR=/usr/local/fsl \
RUN /usr/local/fsl/etc/fslconf/post_install.sh
RUN mkdir -p /etc/fsl
RUN echo "FSLDIR=/usr/local/fsl; . \${FSLDIR}/etc/fslconf/fsl.sh; PATH=\${FSLDIR}/bin:\${PATH}; export FSLDIR PATH" > /etc/fsl/fsl.sh

ENTRYPOINT ["streamlit", "run"]

CMD ["server.py"]
