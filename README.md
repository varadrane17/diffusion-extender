## To run the Extender container 

```bash 
    sudo docker build -f Dockerfile -t extender . && docker system prune -f

    docker run --restart on-failure -d -it -p 8000:8000 --gpus '"device=0"' --name extender1 extender
```
