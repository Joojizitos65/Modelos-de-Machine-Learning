import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    recall_score,
)

RANDOM_STATE = 42
TEST_SIZE = 0.20  # mesma divisao do enunciado: 80/20
ALVO = "Inadimplente"


def separar_treino_teste(dados):
    X = dados.drop(columns=[ALVO])
    y = dados[ALVO]

    # stratify=y mantem a proporcao de inadimplentes igual nos dois conjuntos,
    # importante justamente porque a base e desbalanceada
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return X_tr, X_te, y_tr, y_te


def construir_modelo(class_weight=None, usar_oob=True):
    # os hiperparametros vieram do enunciado. oob_score=True deixa a floresta
    # avaliar cada arvore nas amostras que ficaram de fora do bootstrap dela,
    # entao a gente ganha uma estimativa "de graca" sem gastar o conjunto de teste.
    #
    # na Parte 4 eu desligo o oob (usar_oob=False) porque la nao uso essa metrica,
    # e com os pesos balanceados ela so solta um warning sem serventia.
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        max_features="sqrt",
        oob_score=usar_oob,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        class_weight=class_weight,
    )


# ----------------------------------------------------------------------
# Parte 1 - modelo base
# ----------------------------------------------------------------------
def treinar_modelo_base(dados):
    X_tr, X_te, y_tr, y_te = separar_treino_teste(dados)

    modelo = construir_modelo()
    modelo.fit(X_tr, y_tr)

    # probabilidade da classe 1 (inadimplente) pra calcular o ROC-AUC
    probs = modelo.predict_proba(X_te)[:, 1]

    resultado = {
        "modelo": modelo,
        "X_tr": X_tr,
        "X_te": X_te,
        "y_tr": y_tr,
        "y_te": y_te,
        "oob": modelo.oob_score_,
        "roc_auc": roc_auc_score(y_te, probs),
    }
    return resultado


# ----------------------------------------------------------------------
# Parte 2 - importancia das variaveis
# ----------------------------------------------------------------------
def importancia_variaveis(modelo, colunas):
    # feature_importances_ ja vem como array na ordem das colunas de X.
    # jogo numa Series pra ordenar e ler bonito.
    serie = pd.Series(modelo.feature_importances_, index=colunas)
    return serie.sort_values(ascending=False)


# ----------------------------------------------------------------------
# Parte 3 - analise de desbalanceamento
# ----------------------------------------------------------------------
def proporcao_inadimplentes(dados):
    # media de uma coluna 0/1 ja da direto a fracao de 1s
    return dados[ALVO].mean()


def avaliar_classificacao(modelo, X_te, y_te, threshold=0.5):
    # com threshold 0.5 isso equivale ao predict() normal; deixei parametrizavel
    # porque na Parte 4 vou querer baixar esse corte
    probs = modelo.predict_proba(X_te)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    return {
        "matriz_confusao": confusion_matrix(y_te, y_pred),
        "relatorio": classification_report(y_te, y_pred, zero_division=0),
        "recall_classe1": recall_score(y_te, y_pred, pos_label=1, zero_division=0),
    }


# ----------------------------------------------------------------------
# Parte 4 - desafio: class_weight='balanced' + ajuste de threshold
# ----------------------------------------------------------------------
def treinar_modelo_balanceado(dados, threshold=0.5):
    X_tr, X_te, y_tr, y_te = separar_treino_teste(dados)

    # class_weight='balanced' faz a floresta dar mais peso pros inadimplentes,
    # que sao minoria. assim ela para de "preferir" chutar que todo mundo paga.
    modelo = construir_modelo(class_weight="balanced", usar_oob=False)
    modelo.fit(X_tr, y_tr)

    avaliacao = avaliar_classificacao(modelo, X_te, y_te, threshold=threshold)

    return {
        "modelo": modelo,
        "X_te": X_te,
        "y_te": y_te,
        "threshold": threshold,
        "avaliacao": avaliacao,
    }
