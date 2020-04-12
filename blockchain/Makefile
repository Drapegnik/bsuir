
DIR=fabric-dev-servers
FCARDIR=fabric-samples/fabcar
NETDIR=fabric-samples/first-network
export HL_COMPOSER_CLI=$(PWD)/tutorial-network/node_modules/.bin/composer
export FABRIC_VERSION=hlfv12

node:
	nvm use

##### 1

download-fabric:
	mkdir -p $(DIR)
	cd $(DIR) && curl -O https://raw.githubusercontent.com/hyperledger/composer-tools/master/packages/fabric-dev-servers/fabric-dev-servers.tar.gz
	cd $(DIR) && tar -xvf fabric-dev-servers.tar.gz

download-fabric-runtime:
	cd $(DIR) && ./downloadFabric.sh

start-fabric:
	cd $(DIR) && ./startFabric.sh

create-admin:
	cd $(DIR) && ./createPeerAdminCard.sh

stop-fabric:
	cd $(DIR) && ./stopFabric.sh

clean: stop-fabric
	cd $(DIR) && ./teardownFabric.sh

##### 2

start-fabcar:
	cd $(FCARDIR) && ./startFabric.sh javascript

clean-fabcar:
	cd $(FCARDIR) && ./networkDown.sh

##### 3

generate-network:
	cd $(NETDIR) && ./byfn.sh generate

start-network:
	cd $(NETDIR) && ./byfn.sh up -l javascript

stop-network:
	cd $(NETDIR) && ./byfn.sh down