package main

import (
	"net"
	"sync"

	"github.com/Pallinder/go-randomdata"
)

type storage struct {
	mux        sync.Mutex
	connByName map[string]net.Conn
}

func makeStorage() *storage {
	return &storage{
		connByName: make(map[string]net.Conn),
	}
}

func (s *storage) Save(conn net.Conn) (string, string) {
	ip := conn.RemoteAddr().String()
	name := "@" + randomdata.SillyName()
	s.mux.Lock()
	defer s.mux.Unlock()
	s.connByName[name] = conn
	return name, ip
}

func (s *storage) GetConnection(name string) net.Conn {
	s.mux.Lock()
	defer s.mux.Unlock()
	return s.connByName[name]
}

func (s *storage) GetNames(name string) []string {
	s.mux.Lock()
	names := make([]string, len(s.connByName)-1)
	i := 0
	for n := range s.connByName {
		if n == name {
			continue
		}
		names[i] = n
		i++
	}
	s.mux.Unlock()
	return names
}

func (s *storage) Remove(name string) {
	s.mux.Lock()
	delete(s.connByName, name)
	s.mux.Unlock()
}
