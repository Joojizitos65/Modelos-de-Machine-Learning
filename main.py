from geracao_dados import gerar_base_sintetica
from modelagem_risco import (
    treinar_modelo_base,
    importancia_variaveis,
    proporcao_inadimplentes,
    avaliar_classificacao,
    treinar_modelo_balanceado,
)

LINHA = "=" * 60

# threshold mais baixo pra Parte 4: baixando o corte de 0.50 pra 0.30 a gente
# aceita classificar como inadimplente mais gente, ganhando recall na classe 1
THRESHOLD_AJUSTADO = 0.30


def parte1_modelo_base(dados):
    print(LINHA)
    print("PARTE 1 - MODELO BASE (RandomForest)")
    print(LINHA)

    resultado = treinar_modelo_base(dados)
    print(f"Acuracia OOB (treino): {resultado['oob']:.4f}")
    print(f"ROC-AUC (teste):       {resultado['roc_auc']:.4f}")
    print(
        "\nLeitura: o ROC-AUC mostra que o modelo separa bem quem paga de quem"
        " nao paga, bem melhor do que a acuracia sozinha sugere mais pra frente."
    )
    return resultado


def parte2_importancia(resultado):
    print("\n" + LINHA)
    print("PARTE 2 - IMPORTANCIA DAS VARIAVEIS")
    print(LINHA)

    colunas = resultado["X_tr"].columns
    ranking = importancia_variaveis(resultado["modelo"], colunas)

    for i, (variavel, peso) in enumerate(ranking.items(), start=1):
        print(f"{i}. {variavel:<18} {peso:.4f}")

    campea = ranking.index[0]
    print(f"\nResposta (pergunta 4): a variavel de maior impacto foi '{campea}'.")
    return ranking


def parte3_desbalanceamento(dados, resultado):
    print("\n" + LINHA)
    print("PARTE 3 - ANALISE CRITICA / DESBALANCEAMENTO")
    print(LINHA)

    proporcao = proporcao_inadimplentes(dados)
    print(f"Proporcao de inadimplentes na base: {proporcao:.2%}")
    print(f"Ou seja, cerca de {proporcao:.0%} contra {1 - proporcao:.0%} de adimplentes.")

    avaliacao = avaliar_classificacao(
        resultado["modelo"], resultado["X_te"], resultado["y_te"]
    )
    print("\nMatriz de confusao (linha=real, coluna=previsto):")
    print(avaliacao["matriz_confusao"])
    print("\nRelatorio de classificacao:")
    print(avaliacao["relatorio"])

    print(
        "Resposta (pergunta 7): nao da pra confiar so na acuracia. Como a maioria"
        " dos clientes paga, um modelo que chutasse 'ninguem inadimple' ja acertaria"
        " quase tudo e teria acuracia alta, mas nao pegaria NENHUM caloteiro - que e"
        " exatamente o cliente que a fintech precisa achar. Por isso o olhar tem que"
        " ir pro recall e pro ROC-AUC da classe 1."
    )
    return avaliacao


def parte4_desafio(dados):
    print("\n" + LINHA)
    print("PARTE 4 - DESAFIO: class_weight='balanced' + threshold")
    print(LINHA)

    # recall da classe 1 com o modelo base (threshold padrao 0.50), so pra comparar
    base = treinar_modelo_base(dados)
    recall_base = avaliar_classificacao(base["modelo"], base["X_te"], base["y_te"])[
        "recall_classe1"
    ]

    balanceado = treinar_modelo_balanceado(dados, threshold=THRESHOLD_AJUSTADO)
    avaliacao = balanceado["avaliacao"]

    print(f"Recall da classe 1 (modelo base, threshold 0.50): {recall_base:.2%}")
    print(
        f"Recall da classe 1 (balanced, threshold {THRESHOLD_AJUSTADO:.2f}): "
        f"{avaliacao['recall_classe1']:.2%}"
    )
    print("\nMatriz de confusao do modelo ajustado:")
    print(avaliacao["matriz_confusao"])
    print("\nRelatorio de classificacao do modelo ajustado:")
    print(avaliacao["relatorio"])

    print(
        "Resposta (pergunta 8): usando class_weight='balanced' e baixando o corte"
        " pra 0.30, o modelo passa a pegar bem mais inadimplentes (o recall da classe"
        " 1 sobe bastante). O preco e perder um pouco de precisao - aparecem mais"
        " falsos positivos - mas pra uma fintech deixar passar um caloteiro costuma"
        " custar mais caro do que negar credito pra um bom pagador."
    )


def main():
    print("Gerando base sintetica de clientes...\n")
    dados = gerar_base_sintetica()

    resultado_base = parte1_modelo_base(dados)
    parte2_importancia(resultado_base)
    parte3_desbalanceamento(dados, resultado_base)
    parte4_desafio(dados)

    print("\n" + LINHA)
    print("Fim. Todas as 4 partes rodaram.")
    print(LINHA)


if __name__ == "__main__":
    main()
