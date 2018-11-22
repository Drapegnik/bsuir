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
