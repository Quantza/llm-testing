#!/bin/bash

set -e

yay -Sy
yay -S docker 

sudo systemctl enable docker.service
sudo systemctl start docker.service

sudo docker info

sudo echo Test container...
sudo docker run -it --rm archlinux bash -c "echo hello world"

yay -S docker-compose

sudo usermod -aG docker "$(id -u -n)"
