# The Green Grid — Detecção de Doenças em Plantas

> Projeto desenvolvido para a disciplina **Trabalho Interdisciplinar VI (TI6)**
> Grupo 12 · 2026.1

---

## Apresentações

- [Sprint 1 — Slides (PDF-style)](apresentacoes/sprint1/slides.html)
- [Sprint 2 — Slides (PDF-style)](apresentacoes/sprint2/slides.html)

---

## Problema

Doenças em plantas causam perdas de até 40% nas colheitas mundiais por ano (FAO). O diagnóstico manual depende de agrônomos especializados, recurso escasso em regiões rurais. A detecção tardia agrava o problema.

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
Aquisição → Pré-processamento → Segmentação (ExG) → Treino → Ensemble
```

**Segmentação ExG:** máscara `2G − R − B > 0.1` para isolar a folha do fundo branco uniforme. Mitiga viés de background; o modelo segmented perde cor da lesão mas ganha robustez ao fundo.

**Data augmentation:** flip horizontal, rotação ±15°, ColorJitter — aplicados só no treino para diversificar o domínio sem adicionar dados externos.

### Modelo

ResNet-18 com transfer learning. Backbone ImageNet congelado; apenas a camada FC é treinada (512 → 38 saídas, 19.494 parâmetros treináveis de 11,2M totais).
Hiperparâmetros: `IMG=224 · BATCH=128 · EPOCHS=10 · LR=1e-3 · SEED=42 · NUM_WORKERS=4`
Otimizador: Adam + StepLR (gamma=0.5, step=3 epochs).

### Paralelismo em 3 níveis

```
Nível 3 (distribuído) : Ray actors; 1 actor por versão do dataset, alocados
                        entre nós do cluster (1 ou 2 GPUs)
Nível 2 (paralelo CPU): DataLoader com num_workers=4 processos paralelos de I/O
Nível 1 (paralelo GPU): CUDA; cada batch de 128 imagens processado em paralelo
```

### Estratégias de ensemble

Todos os modelos são avaliados no mesmo `color_test.txt`.

| Estratégia | Descrição | Obs |
|---|---|---|
| Softmax ensemble | Média das probabilidades dos 3 modelos | inclui grayscale fora do domínio de treino |
| Stacking (3 modelos) | LogisticRegression sobre 114 features softmax | inclui grayscale fora do domínio de treino |
| Stacking color+seg | LogisticRegression sobre 76 features (color + segmented) | ambos treinados e avaliados em colorido |

---

## Resultados

### Modelos individuais

Avaliados em `color_test.txt` (10.709 imagens):

| Modelo | Acc | F1 | Precisão | Recall | BalAcc | Obs |
|---|---|---|---|---|---|---|
| **color** | **0.9478** | **0.947** | **0.949** | **0.946** | **0.947** | melhor individual |
| segmented | 0.7703 | 0.766 | | | | perde cor da lesão com ExG |
| grayscale | 0.0890 | 0.021 | | | | colapso por distribution shift |

O modelo grayscale foi treinado em escala de cinza e avaliado em imagens coloridas; o colapso para 8,90% ilustra o risco de distribution shift.

### Ensemble

| Estratégia | Acc | F1 macro | Obs |
|---|---|---|---|
| Softmax ensemble | 0.9279 | 0.9085 | -1.99 pp; grayscale contaminou a média |
| Stacking 3 modelos* | 0.9560 | 0.9434 | +0.82 pp; inclui grayscale fora do domínio |
| **Stacking color+seg** | **0.9530** | **0.9393** | **+0.52 pp sobre o melhor individual** |

**Resposta à pergunta de pesquisa:** o stacking color+segmented supera o melhor modelo individual em +0.52 pp, com ambos os modelos treinados e avaliados em imagens coloridas.

### Escalabilidade

**Forte — DataLoader workers** (60 batches × 128 imgs, benchmark isolado):

| Workers | Tempo (s) | Speedup | Eficiência |
|---|---|---|---|
| 0 (serial) | 15,6 | 1,00× | 100% |
| 1 | 17,1 | 0,91× | 91% |
| 2 | 9,5 | 1,64× | 82% |
| **4** | **7,6** | **2,05×** | **51%** |
| 8 | 8,2 | 1,91× | 24% |

**Forte — Ray cluster** (3 modelos, 10 epochs, mesmo dataset):

| Configuração | Tempo | Speedup | Eficiência |
|---|---|---|---|
| 1 nó (sequencial) | 28,1 min | 1,00× | — |
| **2 nós (paralelo)** | **20,0 min** | **1,41×** | **47%** |

Eficiência 47%: 2 GPUs para 3 actors; 1 actor aguarda GPU livre.

**Fraca — Ray actors:**

| Actors | Trabalho | Tempo/actor | Eficiência fraca |
|---|---|---|---|
| 1 | 1 versão | ~9,4 min | 100% |
| 3 | 3 versões | ~9,4 min | ~100% |

Tarefas completamente independentes (embarrassingly parallel); sem comunicação de gradientes entre actors.

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

**Python 3.12.x** — marque "Add Python to PATH" durante a instalação.

Crie e ative o ambiente virtual:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
```

Instale as dependências:

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

> `--extra-index-url` é obrigatório para o PyTorch com CUDA 12.1. Para outra versão de CUDA, ajuste `cu121` (ex: `cu118`). Verifique com `nvidia-smi`.

Verifique se o CUDA está funcionando:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Estrutura esperada do dataset:**

```
TI6/
├── train.py
├── requirements.txt
├── PlantVillage-completo/
│   ├── raw/
│   │   ├── color/        ← imagens coloridas (38 subpastas por classe)
│   │   ├── grayscale/    ← imagens grayscale
│   │   └── segmented/    ← imagens segmentadas
│   └── splits/
│       ├── color_train.txt
│       ├── color_test.txt
│       ├── grayscale_train.txt
│       ├── grayscale_test.txt
│       ├── segmented_train.txt
│       └── segmented_test.txt
├── checkpoints/          ← criada automaticamente pelo script
└── results/              ← criada automaticamente pelo script
```

> Se os modelos já foram treinados em outro PC, copie também a pasta `checkpoints/` (`model_*_DONE.pt`, `model_*_best.pt`, `stacking_*.pkl`). O script detecta automaticamente e pula o treino, indo direto para avaliação.

### 1 nó (local)

```bash
python train.py
```

### 2 nós (cluster Ray)

```bash
# Variável obrigatória no Windows:
set RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1

# PC1 (head):
ray start --head
python train.py --address=auto

# PC2 (worker) — mesmo comando no PowerShell do PC2:
ray start --address=<IP_DO_PC1>:6379
```

> **Atenção:** o PC2 precisa ter o dataset em `C:\...\PlantVillage-completo\` no mesmo caminho absoluto que o PC1, ou um junction apontando para ele.
### Medir escalabilidade

```bash
# Benchmark DataLoader workers (escalabilidade forte — pré-processamento)
python train.py --benchmark-workers

# Treino sequencial para medir t_seq (baseline)
python train.py --sequential

# Treino paralelo distribuído para medir t_par
python train.py --address=auto
```

Resultados salvos em `checkpoints/timing.json`.

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

### O que o script faz (em ordem)

1. Inicializa o cluster Ray
2. Gera gráficos de EDA (`results/eda_distribuicao.png`, `results/exg_demo.png`)
3. Treina os 3 modelos ResNet-18 em paralelo (color, grayscale, segmented) — ou carrega se já treinados
4. Gera curvas de aprendizado (`results/curvas_aprendizado.png`)
5. Avalia cada modelo em `color_test.txt`
6. Calcula ensemble por média de softmax
7. Treina e avalia stacking (LogisticRegression) — 3 modelos e color+seg
8. Gera heatmap de métricas e gráfico de comparação final
9. Imprime a resposta à pergunta de pesquisa

### Dependências principais

| Pacote | Versão |
|---|---|
| Python | 3.12.x |
| torch (CUDA 12.1) | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| ray | latest |
| numpy | 2.4.3 |
| Pillow | 12.1.1 |
| scikit-learn | 1.6.1 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| joblib | 1.5.0 |

---

## Estrutura do repositório

```
the-green-grid/
├── train.py                   # script principal
├── dataset.py                 # classe PlantVillageDataset (separada para Ray multiprocessing)
├── requirements.txt
├── SETUP.md                   # instruções de ambiente e cluster
├── README.md
├── results/                   # gráficos gerados pelo train.py
├── checkpoints/               # modelos e timing.json (gerados em runtime)
├── apresentacoes/
│   ├── sprint1/               # slides.html (PDF-style)
│   └── sprint2/               # slides.html (PDF-style, Sprint 2)
└── docs/
    └── enunciado.txt
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

1. HUGHES, D. P.; SALATHÉ, M. **An open access repository of images on plant health to enable mobile disease diagnostics.** *arXiv*, 1511.08060, 2015.
2. MOHANTY, S. P.; HUGHES, D. P.; SALATHÉ, M. **Using deep learning for image-based plant disease detection.** *Frontiers in Plant Science*, v. 7, p. 1419, 2016.
3. HE, K. et al. **Deep Residual Learning for Image Recognition.** *IEEE CVPR*, p. 770–778, 2016.
4. SHOAIB, M. et al. **An advanced deep learning models-based plant disease detection: a review of recent research.** *Frontiers in Plant Science*, v. 14, 2023. DOI: 10.3389/fpls.2023.1158933
5. NOYAN, M. A. **Uncovering bias in the PlantVillage dataset.** *arXiv*, 2206.04374, 2022.
6. BARBEDO, J. G. A. **Deep learning applied to plant pathology: the problem of data representativeness.** *Tropical Plant Pathology*, v. 47, p. 85–94, 2022.
7. FAO. **The State of Food and Agriculture 2021.** Food and Agriculture Organization of the United Nations, 2021.

---

> Disciplina: Trabalho Interdisciplinar VI · Grupo 12 · 2026.1
