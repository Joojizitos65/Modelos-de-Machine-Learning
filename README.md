# Modelagem de Risco de Crédito com Random Forest

Atividade prática de Engenharia de Dados: prever quais clientes de uma fintech têm
maior chance de ficar inadimplentes, usando uma base sintética de 2.000 clientes e
um `RandomForestClassifier`.

## Como rodar

```bash
python main.py
```

O `main.py` passa pelas 4 partes da atividade na ordem e já imprime as métricas
junto com uma leitura rápida de cada resultado.

## Organização dos arquivos

- `geracao_dados.py` — gera a base sintética (Parte 1). Renda, Score e Empréstimos
  são sorteados e o rótulo de inadimplência sai de uma função logística.
- `modelagem_risco.py` — treino do modelo, métricas, importância das variáveis e a
  versão balanceada do desafio (Partes 1 a 4).
- `main.py` — amarra tudo e mostra os resultados.

## Resultados que o modelo gera

Rodando com semente 42 (pra todo mundo bater o mesmo número):

| Métrica | Valor |
|---|---|
| Acurácia OOB (treino) | 0,9344 |
| ROC-AUC (teste) | 0,8566 |
| Proporção de inadimplentes na base | 6,75% |
| Recall da classe 1 — modelo base | 7,41% |
| Recall da classe 1 — balanceado + threshold 0,30 | 81,48% |

Importância das variáveis (ordem decrescente):

1. RendaMensal — 0,4121
2. ScoreCredito — 0,3555
3. EmprestimosAtivos — 0,2324

## Respostas das perguntas

### Pergunta 4 — Qual variável teve maior impacto?

A **RendaMensal** foi o fator mais importante, com ~41% de peso na decisão da
floresta, seguida de perto pelo ScoreCredito (~36%). Faz sentido: na fórmula que
gera o rótulo, a renda entra multiplicada por um coeficiente que, somado à faixa
grande de valores que ela assume (de ~1.500 até dezenas de milhares), acaba
mexendo bastante na probabilidade final. O número de empréstimos ativos pesa, mas
varia pouco (quase todo mundo tem 0, 1 ou 2), então contribui menos pra separar os
clientes.

### Pergunta 7 — A acurácia OOB de ~93,4% é confiável pra decisão de negócio?

Não, e esse é o ponto central da atividade. Só 6,75% da base é inadimplente, ou
seja, a classe que a gente mais quer acertar é justamente a minoria. Com esse
desbalanceamento, um modelo "preguiçoso" que simplesmente dissesse que *ninguém*
inadimple já acertaria ~93% dos casos — teria uma acurácia ótima no papel e seria
inútil na prática, porque não pegaria nenhum caloteiro.

É exatamente o que acontece com o modelo base: a acurácia fica em 93%, mas o recall
da classe 1 é de só 7%. Na matriz de confusão dá pra ver que ele deixa passar quase
todos os inadimplentes. Pra decisão de crédito, o que interessa é o **recall da
classe 1** (quantos inadimplentes a gente consegue pegar) e o **ROC-AUC**, que olha
a capacidade de separação independente do threshold. Acurácia sozinha, nesse
cenário, engana.

### Pergunta 8 (desafio) — Impacto do class_weight e do threshold no recall

Com `class_weight='balanced'` a floresta passa a dar mais peso aos inadimplentes na
hora de treinar, e baixando o threshold de decisão de 0,50 para 0,30 a gente aceita
marcar mais clientes como "risco". O efeito nos números:

- Recall da classe 1 pulou de **7,41%** para **81,48%** — o modelo passou a pegar a
  grande maioria dos inadimplentes.
- O preço foi a precisão da classe 1 cair (mais falsos positivos: bons pagadores
  marcados como risco) e a acurácia geral descer pra ~77%.

Pra uma fintech, essa troca costuma valer a pena: liberar crédito pra quem vai dar
calote (falso negativo) geralmente custa mais caro do que negar/pedir garantia a um
bom pagador (falso positivo). Ou seja, perder um pouco de acurácia global pra ganhar
muito recall na classe que importa é um bom negócio aqui.
