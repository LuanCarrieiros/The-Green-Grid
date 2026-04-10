# The Green Grid — Detecção de Doenças em Plantas

> Projeto desenvolvido para a disciplina **Trabalho Interdisciplinar VI (TI6)**
> Grupo 12 · 2026.1

---

## Apresentações

- [Sprint 1 — PDF](https://github.com/LuanCarrieiros/The-Green-Grid/blob/main/apresentacoes/sprint1/sprint1.pdf)
- [Sprint 2 — Apresentação interativa](https://luancarrieiros.github.io/The-Green-Grid/apresentacoes/sprint2/apresentacao.html)

---

## Problema

Doenças em plantas causam perdas de até 40% nas colheitas mundiais por ano (FAO). O diagnóstico manual depende de agrônomos especializados — mão de obra cara e escassa em regiões rurais. A detecção tardia agrava o problema.

## Objetivo

Desenvolver um pipeline de Processamento e Análise de Imagens capaz de:

- Classificar imagens de folhas em 38 categorias (doenças + saudável)
- Treinar 3 modelos em paralelo via computação distribuída (Ray)
- Comparar estratégias de ensemble para maximizar a acurácia

---

## Dataset — PlantVillage

Base open source criada pelo **Digital Epidemiology Lab, EPFL** (Hughes & Salathé, 2015).
Disponível no HuggingFace: `mohanty/PlantVillage`

| Atributo | Valor |
|---|---|
| Total de imagens | 54.306 |
| Classes | 38 (26 doenças + 12 categorias saudáveis) |
| Culturas | 14 |
| Versões | color · grayscale · segmented |
| Split | 80/20 (43.596 treino / ~10.700 teste por versão) |
| Licença | CC-BY-SA-3.0 |

---

## Metodologia

### Pipeline

```
Aquisição → Pré-processamento → Segmentação (ExG) → Classificação
```

### Modelo

ResNet-18 com transfer learning — apenas a camada FC é treinada (38 saídas).
Hiperparâmetros: `IMG=224 · BATCH=128 · EPOCHS=10 · LR=1e-3 · SEED=42`

### Paralelismo em 3 níveis

```
Nível 3 (distribuído) : Ray actors — 1 actor por versão do dataset, alocados
                        entre nós do cluster (1 ou 2 PCs)
Nível 2 (paralelo CPU): DataLoader com num_workers=4 processos paralelos de I/O
Nível 1 (paralelo GPU): CUDA — cada batch processado em paralelo nos cores da GPU
```

### Estratégias de ensemble

Todos os modelos são avaliados no mesmo `color_test.txt` (comparação justa).

| Estratégia | Descrição | Justa? |
|---|---|---|
| Softmax ensemble | Média das probabilidades dos 3 modelos | Sim |
| Stacking (3 modelos) | LogisticRegression sobre 114 features softmax | Não* |
| Stacking color+seg | LogisticRegression sobre 76 features (color + segmented) | Sim |

\* *grayscale e segmented recebem imagens coloridas, fora da sua distribuição de treino*

---

## Resultados

Avaliados em `color_test.txt` (10.709 imagens):

| Modelo | Acc | F1 macro | Obs |
|---|---|---|---|
| color | **0.9478** | 0.9328 | melhor individual |
| segmented | 0.7703 | 0.6879 | degradou com input colorido |
| grayscale | 0.0890 | 0.0444 | colapso — distribution shift |
| Softmax ensemble | 0.9279 | 0.9085 | -1.99 pp — grayscale contaminou a média |
| Stacking 3 modelos* | 0.9560 | 0.9434 | +0.82 pp — comparação não justa |
| **Stacking color+seg** | **0.9530** | **0.9393** | **+0.52 pp — comparação justa** |

**Resposta à pergunta de pesquisa:** o stacking color+segmented supera o melhor modelo individual em +0.52 pp numa comparação justa.

Gráficos gerados em `results/`:

| Arquivo | Conteúdo |
|---|---|
| `comparacao_final.png` | Acurácia de todos os modelos e ensembles |
| `curvas_aprendizado.png` | Loss e acurácia de validação por epoch |
| `metricas_heatmap.png` | Heatmap de métricas (acc, f1, prec, rec, bal_acc) |
| `matriz_confusao_ensemble.png` | Matriz de confusão do softmax ensemble (38×38) |
| `eda_distribuicao.png` | Distribuição de imagens por cultura |
| `exg_demo.png` | Demonstração do filtro Excess Green (ExG) |

---

## Como rodar

### Pré-requisitos

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

O dataset deve estar em `PlantVillage-completo/` (baixar via HuggingFace).

### 1 nó (local)

```bash
python train.py
```

### 2 nós (cluster Ray)

```bash
# No PC1 (head):
ray start --head
python train.py --address=auto

# No PC2 (worker):
ray start --address=<IP_DO_PC1>:6379
```

### Checkpoints

O script salva o estado ao final de cada epoch — se interrompido, retoma de onde parou.

```
checkpoints/
  model_{ver}_best.pt       → melhor epoch (por val_acc)
  model_{ver}_DONE.pt       → treino completo + histórico
  model_{ver}_resume.pt     → estado atual (existe só durante treino)
  stacking_meta.pkl         → resultado do stacking 3 modelos
  stacking_model.pkl        → objeto LogisticRegression (3 modelos)
  stacking_cs_meta.pkl      → resultado do stacking color+seg
  stacking_cs_model.pkl     → objeto LogisticRegression (color+seg)
```

---

## Estrutura do repositório

```
the-green-grid/
├── train.py                   # script principal
├── green_grid.ipynb           # notebook (treino inicial, num_workers=0)
├── requirements.txt
├── README.md
├── results/                   # gráficos gerados pelo train.py
├── apresentacoes/
│   ├── sprint1/               # slides.html + PDFs do Sprint 1
│   └── sprint2/               # apresentacao.html (Reveal.js)
└── docs/
    ├── enunciado.txt
    ├── meutexto.txt
    └── oquefiz.txt
```

---

## Equipe

Arthur Clemente Machado  
Diego Moreira Rocha  
Felipe Vilhena Dias  
Iago Fereguetti Ribeiro  
Luan Barbosa Rosa Carrieiros  
Lucas Henrique Rocha Hauck

---

## Referências

1. HUGHES, D. P.; SALATHÉ, M. **An open access repository of images on plant health to enable the development of mobile disease diagnostics.** *arXiv*, 1511.08060, 2015.
2. MOHANTY, S. P.; HUGHES, D. P.; SALATHÉ, M. **Using deep learning for image-based plant disease detection.** *Frontiers in Plant Science*, v. 7, p. 1419, 2016.
3. HE, K. et al. **Deep Residual Learning for Image Recognition.** *IEEE CVPR*, 2016.
4. FERENTINOS, K. P. **Deep learning models for plant disease detection and diagnosis.** *Computers and Electronics in Agriculture*, v. 145, p. 311–318, 2018.
5. MORITZ, P. et al. **Ray: A Distributed Framework for Emerging AI Applications.** *USENIX OSDI*, 2018.
6. GONZALEZ, R. C.; WOODS, R. E. **Processamento Digital de Imagens.** 3. ed. Pearson, 2010.
7. THAPA, R. et al. **The Plant Pathology Challenge 2020 data set to classify foliar disease of apples.** *Applications in Plant Sciences*, 2020.
8. DOSOVITSKIY, A. et al. **An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale.** *ICLR*, 2021.
9. BARBEDO, J. G. A. **Data fusion in agriculture: resolving ambiguities and closing data gaps using multi-resolution, multi-temporal and multi-spectral satellite imagery.** *Remote Sensing*, v. 14, n. 19, 2022.

---

> Disciplina: Trabalho Interdisciplinar VI · Grupo 12 · 2026.1
