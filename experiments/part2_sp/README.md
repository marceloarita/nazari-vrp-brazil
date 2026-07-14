# Part II — São Paulo

Trabalho a partir da tag `v1.0-nazari-reproduction` (Part I: reprodução do Nazari + zero-shot).
Foco: fechar o gap de distribuição em SP treinando **por zona**, casando treino e eval.

## Escopo (4 alavancas)
- **A1 — Dados**: KDE per-cluster do zero (1 zona)
- **A2 — Features**: densidade KDE por nó → `[x, y, demanda, log p(x,y)]`
- **A3 — Baseline**: POMO (média de N rollouts multi-start; substitui o Kool, sem critic)
- **A4 — Decoding**: 2-opt pós-processamento + sampling no eval

## Convenção de experimentos
```
{size}_{dist}_{zone}__{baseline}__{features}
ex.: vrp20_kde_c1__pomo__dens
```
- Checkpoints: `artifacts/checkpoints/<exp>/`
- Resultados:  `artifacts/results/<...>.csv` (coluna `exp` = ID)
- Instâncias:  `artifacts/instances/`

`notes.md` neste diretório é um rascunho vivo (não versionado).
