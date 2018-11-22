package main

import (
	"bufio"
	"fmt"
	"net"
)

func handleConnection(conn net.Conn, store *storage) {
	name, ip := store.Save(conn)
	log.info("new connection - %s (%s)", name, ip)
	sendMessage(conn, fmt.Sprintf("Hello, %s!", name), SYSTEM)

	scanner := bufio.NewScanner(conn)
	for {
		ok := scanner.Scan()
		if !ok {
			break
		}
		action := getAction(scanner.Bytes())
		switch action.Type {
		case CLIENTS_REQUEST:
			clients := store.GetNames(name)
			log.info("%s requested clients: %q", name, clients)
			send(conn, stringify(makeAction(CLIENTS_RESPONSE, clients, SYSTEM)))
		case MESSAGE_REQUEST:
			toConn := store.GetConnection(action.To)
			message := action.Payload.(string)
			sendMessage(toConn, message, name)
			log.info("%s -> %s: %s", name, action.To, message)
		default:
			log.warning("unexpected request action type: %s", action.Type)
		}
	}
	log.info("%s disconnected", name)
	store.Remove(name)
}

func server(addr string) {
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		log.fatal(err)
	}
	defer listener.Close()
	log.info("start tcp server at %s", addr)

	store := makeStorage()
	for {
		conn, err := listener.Accept()
		if err != nil {
			log.error(err)
			continue
		}
		go handleConnection(conn, store)
	}
}
