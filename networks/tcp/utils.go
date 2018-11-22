package main

import (
	"fmt"
	"os"
)

func getPrefix(level string) string {
	return fmt.Sprintf("[%s]: \t", level)
}

func print(level string, format string, a ...interface{}) {
	fmt.Printf("%s%s\n", getPrefix(level), fmt.Sprintf(format, a...))
}

func printMessage(action *Action) {
	fmt.Printf("> (%s): %s\n", action.Author, action.Payload)
}

type logger struct {
}

func (l *logger) info(format string, a ...interface{}) {
	print("info", format, a...)
}

func (l *logger) warning(format string, a ...interface{}) {
	print("warning", format, a...)
}

func (l *logger) fatal(err interface{}) {
	fmt.Printf("%s%s\n", getPrefix("fatal"), err)
	os.Exit(1)
}

func (l *logger) error(err interface{}) {
	fmt.Printf("%s%s\n", getPrefix("error"), err)
}

var log = logger{}
