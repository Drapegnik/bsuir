# ml help

## conv

каждый фильтр свёртки извлекает какую-нибудь фичу

- lenet: `6*10e4` (60 тыс)
- alex: `6*10e6` (60 млн)
- vgg: `13,8*10e6` (138 млн)

> vgg = 2 alex = 100 lenet

## cross ent

- `categorical_crossentropy` - one-hot encoding
- `sparse_categorical_crossentropy` - integers

## rnn
### lstm

- долгосрочная память
- на предыдущем шаге

- forget gate: sigmoid: решает сколько информации из долгосрочной пропустить дальше
- input gate: tann: сколько дальше из текущей
- out gate: сколько дальше на вход

`C(t) = f * C(t-1) + i * C(t)`


In the case of GloVe, the counts matrix is preprocessed by normalizing the counts and log-smoothing them.
Compared to word2vec, GloVe allows for parallel implementation

which contains the information on how frequently each “word” (stored in rows), is seen in some “context” (the columns)
