package main

import (
	"flag"
	"fmt"
	"os"
	"strconv"
)

var host = flag.String("host", "localhost", "The address to listen on")
var port = flag.Int("port", 8000, "The port to listen on")
var mode = flag.String("mode", "server", "The mode [server|client]")

func main() {
	flag.Parse()
	addr := *host + ":" + strconv.Itoa(*port)

	switch *mode {
	case "server":
		server(addr)
	case "client":
		client(addr)
	default:
		fmt.Printf("unexpected value %s for -mode flag\n", *mode)
		flag.PrintDefaults()
		os.Exit(2)
	}
}
