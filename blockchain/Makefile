
DIR=fabric-dev-servers
export HL_COMPOSER_CLI=$(PWD)/tutorial-network/node_modules/.bin/composer
export FABRIC_VERSION=hlfv12

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