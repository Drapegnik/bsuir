package main

import (
	"fmt"
	"os"
)

func getPrefix(level string) string {
	return fmt.Sprintf("\n> [%s]:\t", level)
}

type logger struct {
}

func (l *logger) info(format string, a ...interface{}) {
	fmt.Printf("%s%s\n", getPrefix("info"), fmt.Sprintf(format, a...))
}

func (l *logger) fatal(err interface{}) {
	fmt.Printf("%s%s\n", getPrefix("fatal"), err)
	os.Exit(1)
}

func (l *logger) error(err interface{}) {
	fmt.Printf("%s%s\n", getPrefix("error"), err)
}

var log = logger{}
