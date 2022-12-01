FROM python:3.10.1-slim

EXPOSE 8501

WORKDIR /DS1016-Project

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/Shaurynn/DS1016-Project.git .

RUN pip3 install -r requirements.txt

RUN "python /tmp/fslinstaller.py -D -E -d /usr/local/fsl"

ENTRYPOINT ["streamlit", "run", "server.py", "--server.port=8501", "--server.address=0.0.0.0"]
