package main

import (
	"bufio"
	"net"
	"os"
)

func handleAction(action *Action) {
	switch action.Type {
	case MESSAGE:
		printMessage(action)
	default:
		log.warning("unexpected action type: %s", action.Type)
	}
}

func listen(conn net.Conn) {
	scanner := bufio.NewScanner(conn)
	for {
		ok := scanner.Scan()
		if !ok {
			log.info("disconnected from server")
			log.warning("shutting down...")
			os.Exit(1)
			break
		}
		action := getAction(scanner.Bytes())
		handleAction(action)
	}
}

func client(addr string) {
	log.info("connecting to %s...", addr)
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		log.fatal(err)
	}
	defer conn.Close()
	log.info("successfully connected")

	go listen(conn)

	for {
	}
}
