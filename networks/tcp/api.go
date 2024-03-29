package main

import (
	"encoding/json"
	"net"
)

type actionType string

const (
	// SYSTEM username
	SYSTEM           string     = "@system"
	MESSAGE          actionType = "MESSAGE"
	MESSAGE_REQUEST  actionType = "MESSAGE_REQUEST"
	CLIENTS_REQUEST  actionType = "CLIENTS_REQUEST"
	CLIENTS_RESPONSE actionType = "CLIENTS_RESPONSE"
)

// Action is serializable communication message format
type Action struct {
	Type    actionType  `json:"type"`
	Payload interface{} `json:"payload"`
	Author  string      `json:"author"`
	To      string      `json:"to"`
}

func makeAction(aType actionType, payload interface{}, author string) *Action {
	return &Action{
		Type:    aType,
		Payload: payload,
		Author:  author,
	}
}

func makeRequestAction(aType actionType, payload interface{}, to string) *Action {
	return &Action{Type: aType, Payload: payload, To: to}
}

func stringify(v interface{}) string {
	data, err := json.Marshal(v)
	if err != nil {
		log.error(err)
		return ""
	}
	return string(data) + "\n"
}

func getAction(data []byte) *Action {
	action := &Action{}
	err := json.Unmarshal(data, action)
	if err != nil {
		log.error(err)
	}
	return action
}

func sendMessage(to net.Conn, text string, from string) {
	send(to, stringify(makeAction(MESSAGE, text, from)))
}

func send(conn net.Conn, data string) {
	_, err := conn.Write([]byte(data))
	if err != nil {
		log.error(err)
	}
}
