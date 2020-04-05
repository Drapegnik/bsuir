'use strict';

const fs = require('fs');
const path = require('path');

// load the network configuration
const ccpPath = path.resolve(
    __dirname,
    '..',
    'fabric-samples',
    'test-network',
    'organizations',
    'peerOrganizations',
    'org1.example.com',
    'connection-org1.json'
);
const ccp = JSON.parse(fs.readFileSync(ccpPath, 'utf8'));

module.exports = {ccp};