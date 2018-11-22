# TCP server

Использование сетевых ресурсов: сервер сообщений

## task

### server

Создать многопоточный TCP-сервер со следующим функционалом:

- фиксировать все попытки входящих соединений со стороны клиентов
- сервер должен идентифицировать клиента по его `ip`-адресу и сопоставлять с ним имя (к клиенту в дальнейшем можно обращаться по имени или по `ip`-адресу)
- организовать постоянный приём и отсылку сообщений от клиентов (каждый клиент должен иметь возможность отправить любому другому присоединенному к серверу клиенту сообщение), сообщения оформляются как команды с помощью `JSON`
- по требованию клиента сервер должен выдавать список всех присоединенных к нему клиентов (список оформляется с помощью `JSON`)

### client

Организовать отсылку команд серверу. Команды вводятся пользователем с клавиатуры.

## setup & run

1. install [golang](https://golang.org/doc/install)
3. `sh setup.sh`
4. run server with `./run-server.sh`
5. run clients with `./run-client.sh`

## solution

### server

[![Image from Gyazo](https://i.gyazo.com/2497a5d64e3a280cb7be2274ff3cc835.gif)](https://gyazo.com/2497a5d64e3a280cb7be2274ff3cc835)

### client

[![Image from Gyazo](https://i.gyazo.com/7d2bf36b285b3b527f5d047c3245dbc7.gif)](https://gyazo.com/7d2bf36b285b3b527f5d047c3245dbc7)

- to verify that `server.go` serves multiple TCP clients 
```bash
$ netstat -anp TCP | grep 8000
```

- to test with `nc`:
```bash
$ nc localhost 8000
{"type": "ping", "payload": "hello"}
```
