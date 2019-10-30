# sqa - software quality assurance
Обеспечение качества программного обеспечения

## task
1. Написать калькулятор, который работает на `lambda-functions` в **AWS**.
  - Программа предоставляет простое API по базовым математическим операциям над массивами (например, сложение массивов, умножение массива на скаляр).
  - Подключить **Swagger**. UI не нужен.
  - Программа экспериментальная, не надо на ней делать акцент - несколько базовых, но ресурсоемких операций хватит.
2. Подключить **CloudWatch** и собирать ошибки, а также `info` и `warning` логи в нем
3. Подключить **X-Ray** - разметить код тегами и логировать вызовы
4. Сделать нагрузочное тестирование - послать миллионы запросов и наблюдать за выполнением. Собрать в удобном для анализа метрики производительности.
5. Сделать `load balancing` нативными средствами **AWS**.
6. Повторить нагрузочный тест и проанализировать результаты

## dev

- `make install`
- `make test`
- `make build` - build & pack aws layer

> use `drapegnik-Python37-numpy-aws-x-ray-sdk.zip` for `numpy` & `aws-xray-sdk` dependencies

> check out [Creating New AWS Lambda Layer](https://medium.com/@qtangs/creating-new-aws-lambda-layer-for-python-pandas-library-348b126e9f3e) & [Layers Docs](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html)

### api
```bash
curl -X POST https://fkjl4ua0b7.execute-api.us-east-1.amazonaws.com/default/calculator \
-H "Content-Type: application/json" \
-d @- << EOF
{
  "op": "+",
  "a": [1, 2, 3],
  "b": [3, 2, 1]
}
EOF

# -> {"result": [4, 4, 4]}
```

### load balancer
```bash
curl -X POST http://calculator-load-balancer-315442892.us-east-1.elb.amazonaws.com \
-H "Content-Type: application/json" \
-d @- << EOF
{
  "op": "*",
  "a": [1, 2, 3],
  "b": [3, 2, 1]
}
EOF

# -> {"result": [3, 4, 3]}
```
