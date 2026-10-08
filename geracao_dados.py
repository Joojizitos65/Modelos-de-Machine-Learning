import numpy as np
import pandas as pd

# semente fixa pra todo mundo gerar exatamente a mesma base e os numeros baterem
SEMENTE = 42
N_CLIENTES = 2000


def gerar_base_sintetica(n=N_CLIENTES, semente=SEMENTE):
    np.random.seed(semente)

    dados = pd.DataFrame(
        {
            # exponential + 1500 pra ninguem ter renda perto de zero
            "RendaMensal": np.random.exponential(5000, n) + 1500,
            "ScoreCredito": np.random.randint(300, 950, n),
            "EmprestimosAtivos": np.random.poisson(1.5, n),
        }
    )

    # probabilidade de inadimplencia via funcao logistica
    prob = 1 / (
        1
        + np.exp(
            -(
                -0.0003 * dados["RendaMensal"]
                - 0.005 * dados["ScoreCredito"]
                + 0.8 * dados["EmprestimosAtivos"]
            )
        )
    )

    # vira rotulo 0/1: se a prob supera o sorteio, o cliente inadimple
    dados["Inadimplente"] = (prob > np.random.uniform(0, 1, n)).astype(int)

    return dados


if __name__ == "__main__":
    base = gerar_base_sintetica()
    print("Primeiras linhas da base:")
    print(base.head(10))
    print("\nDistribuicao de inadimplentes:")
    print(base["Inadimplente"].value_counts())
