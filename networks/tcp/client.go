package main

import (
	"bufio"
	"errors"
	"net"
	"os"
	"time"

	"github.com/manifoldco/promptui"
)

const (
	SEND_MESSAGE string = "send message to"
	QUIT         string = "quit"
)

func listen(conn net.Conn, ch chan []interface{}) {
	scanner := bufio.NewScanner(conn)
	for {
		ok := scanner.Scan()
		if !ok {
			log.info("disconnected from server")
			os.Exit(0)
		}
		action := getAction(scanner.Bytes())
		switch action.Type {
		case MESSAGE:
			printMessage(action)
		case CLIENTS_RESPONSE:
			ch <- action.Payload.([]interface{})
		default:
			log.warning("unexpected action type: %s", action.Type)
		}
	}
}

func selectRecipient(clients []interface{}) string {
	clientsPrompt := promptui.Select{
		Label: "Select Recipient",
		Items: clients,
	}

	_, sendTo, err := clientsPrompt.Run()

	if err != nil {
		log.fatal(err)
	}

	return sendTo
}

func promptMessage() string {
	validate := func(input string) error {
		if len(input) < 1 {
			return errors.New("Message cant be empty")
		}
		return nil
	}

	messagePromt := promptui.Prompt{
		Label:    "Message: ",
		Validate: validate,
	}

	message, err := messagePromt.Run()

	if err != nil {
		log.fatal(err)
	}

	return message
}

func client(addr string) {
	log.info("connecting to %s...", addr)
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		log.fatal(err)
	}
	defer conn.Close()
	log.info("successfully connected")

	clientsChannel := make(chan []interface{})
	go listen(conn, clientsChannel)
	time.Sleep(time.Millisecond)

	prompt := promptui.Select{
		Label: "Select Action",
		Items: []string{SEND_MESSAGE, QUIT},
	}

	for {
		_, result, err := prompt.Run()

		if err != nil {
			log.fatal(err)
		}

		switch result {
		case SEND_MESSAGE:
			send(conn, stringify(makeRequestAction(CLIENTS_REQUEST, "", SYSTEM)))
			clients := <-clientsChannel
			if len(clients) == 0 {
				log.warning("chat empty")
				continue
			}
			recipient := selectRecipient(clients)
			message := promptMessage()
			send(conn, stringify(makeRequestAction(MESSAGE_REQUEST, message, recipient)))
		case QUIT:
			log.info("bye!")
			return
		}
	}
}
