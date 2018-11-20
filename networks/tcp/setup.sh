#!/usr/bin/env bash

# install all dependencies
go get -d ./...

# make this files executable
chmod +x run-client.sh
chmod +x run-server.sh
