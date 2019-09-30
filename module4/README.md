# Systèmes, paradigmes et langages pour les Big Data

## Setup to run the notebooks

You will run the exercises on your laptop using a docker container. You should
1. Install docker [link for Windows](https://docs.docker.com/docker-for-windows/install/)
2. Download and run at least once the docker image [jupyter/pyspark-notebook](https://hub.docker.com/r/jupyter/pyspark-notebook), [documentation](https://jupyter-docker-stacks.readthedocs.io/en/latest/)
    - On a Windows laptop with Docker desktop community installed and an existing folder C:\work, you can run the following commands in powershell
    - `docker pull jupyter/pyspark-notebook`
    - `docker run --mount src="C:\work ",target=/home/jovyan/work,type=bind -p 8888:8888 -p 4040:4040 jupyter/pyspark-notebook`
    - Copy paste the url written in powershell in your browser http://127.0.0.1:8888/?token=125765367854365
3. You are good if you see the jupyter home page in your browser 

