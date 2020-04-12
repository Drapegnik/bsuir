# blockchain

1. [Hyperledger Composer Developer Tutorial](https://hyperledger.github.io/composer/latest/tutorials/developer-tutorial)
2. [Writing Your First Application](https://hyperledger-fabric.readthedocs.io/en/latest/write_first_app.html#writing-your-first-application)
3. [Building Your First Network](https://hyperledger-fabric.readthedocs.io/en/latest/build_network.html)

## fabric scripts

- `make start-fabric`

- `make create-admin`

- `make stop-fabric`

- `make start-fabcar`

- `make clean-fabcar`

## 1. tutorial-network

- `make server`

- `make playground`

- `make client`

## 2. fabcar app

> Require [`fabric-samples`](https://github.com/hyperledger/fabric-samples) to be placed in the current directory

> Check out [Installation Guide](https://hyperledger-fabric.readthedocs.io/en/latest/install.html)

- `make logs`

- `make admin`

- `make user`

- `make query`

- `make update`

## 3. first-network

- `make generate-network`

- `make start-network`

- `make stop-network`

#### underhood

- `$ cryptogen generate --config=./crypto-config.yaml`

- `$ configtxgen -profile SampleMultiNodeEtcdRaft -channelID byfn-sys-channel -outputBlock ./channel-artifacts/genesis.block`

- `$ CHANNEL_NAME=mychannel  && configtxgen -profile TwoOrgsChannel -outputCreateChannelTx ./channel-artifacts/channel.tx -channelID $CHANNEL_NAME`

- `$ docker-compose -f docker-compose-cli.yaml -f docker-compose-etcdraft2.yaml up -d`