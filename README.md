## To run the Extender container 

```bash 
    sudo docker build -f Dockerfile -t extender . && docker system prune -f

    docker run --restart on-failure --volume $PWD:/images -d -it -p 8000:8000 --gpus '"device=0"' --name extender1 extender
    docker run --restart on-failure --volume $PWD/images:/storage -d -it -p 8110:8000 -p 3265:3265 --gpus '"device=2"' --name extender-dev extender_dev 
```
