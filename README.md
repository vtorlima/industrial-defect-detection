# Detecção de Defeitos Industriais com Processamento Digital de Imagens 

Detecção, segmentação e classificação de defeitos em imagens industriais do dataset **MVTec AD**, usando apenas técnicas clássicas de Processamento Digital de Imagens (PDI) e atributos interpretáveis, **sem redes neurais** em nenhuma etapa.

## Integrantes da Equipe

| Nome | RA |
|---|---|
| Danilo Djovano Selli Junior | 842868 |
| Vitor Lima dos Santos | 845564 |

## Questão do projeto

> É possível detectar e classificar defeitos industriais usando somente PDI clássico e atributos interpretáveis?

Para responder, o projeto constrói um pipeline completo de ponta a ponta (localização do defeito, extração de atributos e classificação) e mede o efeito de cada etapa por meio de experimentos comparativos.

## Dataset e categoria escolhida

- **Dataset:** [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- **Licença:** CC BY-NC-SA 4.0 (uso não comercial, com atribuição e compartilhamento pela mesma licença)
- **Categoria:** `bottle`

**Por que `bottle`?**

- A garrafa aparece vista de cima, centrada e com simetria circular, então uma imagem de referência média funciona sem registro complexo.
- O fundo é uniforme.
- Os três tipos de defeito separam bem as famílias de atributos: `broken_large` e `broken_small` alteram a **geometria**, enquanto `contamination` altera **intensidade/textura**.
- Ponto fraco: poucas imagens defeituosas (cerca de 63), o que exige validação cruzada.

| Tipo de imagem | Quantidade (aprox.) |
|---|---|
| Treino (boas) | 209 |
| Teste (boas) | 20 |
| Teste (defeituosas) | 63 |

> As contagens são aproximadas, conforme o MVTec AD. Confirmar ao baixar.

## Pipeline

```
Imagem → Redimensionamento (256×256) → Denoising → ROI (Hough) → Mapa de anomalia (referência)
       → Segmentação (A: diferença p/ referência | B: K-means em Lab)
       → Morfologia (abertura, fechamento, remoção de componentes pequenos)
       → Extração de atributos (geometria, intensidade/cor, textura, contorno/HOG)
       → Classificação (SVM | Random Forest)
```

### Etapas e técnicas

| Etapa | Técnicas |
|---|---|
| Pré-processamento | Redimensionamento 256×256 (`INTER_AREA` nas imagens, `INTER_NEAREST` nas máscaras), filtros Gaussiano, mediana e bilateral |
| ROI | `cv2.HoughCircles` para localizar a boca da garrafa |
| Referência estatística | Média μ(x,y) e desvio σ(x,y) por pixel (cinza e Lab) |
| Mapa de anomalia | `z(x,y) = abs(I(x,y) − μ(x,y)) / (σ(x,y) + ε)`, vetorizado com NumPy |
| Bordas | Sobel (diferença de gradiente) e Canny (bordas novas em relação à referência dilatada) |
| Segmentação | **Método A:** diferença para referência com limiar calibrado (variante Otsu). **Método B:** K-means em Lab + textura local |
| Morfologia | Abertura, fechamento, remoção de componentes por área (`cv2.connectedComponentsWithStats`) |
| Atributos | Geometria, intensidade/cor, textura (LBP, GLCM), contorno (curvatura, momentos de Hu, HOG reduzido) |
| Classificação | SVM (kernel RBF) e Random Forest, ambos com `class_weight="balanced"` |

## Partição dos dados

No MVTec AD original, as imagens defeituosas só existem no conjunto de teste (protocolo não supervisionado). Por isso, usamos uma **partição supervisionada própria**:

| Conjunto | Conteúdo | Uso |
|---|---|---|
| Referência | ~120 boas do treino | Imagem média e desvio por pixel |
| Calibração | ~30 boas do treino | Escolha dos limiares da segmentação |
| Desenvolvimento | 3 defeituosas de cada tipo (9) | Ajuste de parâmetros; fora das métricas finais |
| Pool supervisionado | 59 boas do treino + 20 boas do teste + 54 defeituosas | Métricas de segmentação e validação cruzada |

**Regras contra vazamento de dados:**

- Imagens de referência e calibração nunca entram no pool.
- `StratifiedKFold` com 5 folds, estratificado por tipo, repetido com 3 a 5 sementes (resultados em média ± desvio).
- Hiperparâmetros escolhidos apenas com os folds de treino.
- O `StandardScaler` fica dentro do `Pipeline` do scikit-learn.

> **Aviso:** os números deste projeto vêm de um experimento didático com partição própria e **não são comparáveis ao benchmark oficial do MVTec AD**.

## Atributos extraídos

Os atributos são calculados sobre a **máscara prevista**, nunca sobre a máscara de ground truth (exceto no teste do oráculo).

| Grupo | Atributos |
|---|---|
| `geo_` Geometria | Área, perímetro, extensão, orientação, compacidade, excentricidade, solidez, nº de componentes, número de Euler |
| `int_` Intensidade/cor | Média e desvio por canal Lab, contraste região × anel, histograma de 8 faixas |
| `tex_` Textura | LBP uniforme (P=8, R=1), GLCM (contraste, homogeneidade, energia, correlação), entropia, uniformidade |
| `cnt_` Contorno/HOG | Curvatura média, nº de concavidades, 7 momentos de Hu, HOG reduzido |

## Métricas

- **Segmentação** (implementadas à mão com NumPy a partir de TP, FP e FN): Dice, IoU, precisão e recall, por imagem e agregadas. Nas imagens boas, usa-se a taxa de pixels falsos positivos e o número de componentes espúrios.
- **Classificação** (defeituosa = classe positiva): F1, balanced accuracy e ROC-AUC, com média ± desvio sobre folds e sementes.

## Experimentos

| # | Experimento | Pergunta |
|---|---|---|
| 1 | Duas estratégias de segmentação | Diferença para referência (A) × K-means (B) |
| 2 | Efeito da morfologia | Como abertura/fechamento afetam falsos positivos e o recall de defeitos pequenos? |
| 3 | Grupos de atributos | Geometria × textura × combinação completa |
| 4 | Classificadores | SVM × Random Forest na mesma partição |
| 5 | Resultados por tipo de defeito | Qual tipo é mais difícil e por quê (tamanho, contraste ou ambos)? |

### Análises de gargalo

1. **Oráculo de segmentação:** classificação com atributos calculados na máscara GT.
2. **Rastreio de falsos negativos:** defeito perdido na segmentação ou erro do classificador?
3. **Ablação por etapa:** remoção de uma etapa por vez (denoising, bordas, morfologia) e medição de ΔDice e ΔF1.
4. **Troca de classificador:** SVM e RF empatando indica que o classificador não é o gargalo.

## Estrutura sugerida do repositório

```
.
├── src/
│   ├── io.py
│   ├── preprocess.py
│   ├── reference.py
│   ├── segment.py
│   ├── morph.py
│   ├── metrics.py
│   ├── features.py
│   └── classify.py
├── notebooks/        # um notebook por experimento
├── results/          # CSVs com resultados e parâmetros de cada rodada
├── data/             # MVTec AD (categoria bottle), não versionado
└── README.md
```

## Tecnologias

- Python
- OpenCV
- scikit-image
- scikit-learn
- NumPy
- Matplotlib

## Como executar

1. Baixe a categoria `bottle` do MVTec AD e extraia em `data/`.
2. Instale as dependências:
   ```bash
   pip install numpy opencv-python scikit-image scikit-learn matplotlib
   ```
3. Execute os notebooks da pasta `notebooks/` na ordem dos experimentos (também é possível rodar no Google Colab).

## Cronograma

| Semanas | Fases |
|---|---|
| 1 a 2 | Preparação, exploração e partição (Fases 0 a 2) |
| 2 a 3 | Referência e baseline, métricas (Fases 3 e 7). **Marco M1: baseline completo** |
| 4 | Denoising e bordas (Fase 4) |
| 5 a 6 | Segmentação e morfologia (Fases 5 e 6). **Marco M2: Experimentos 1 e 2** |
| 7 a 8 | Atributos e classificação (Fases 8 e 9). **Marco M3: Experimentos 3 e 4** |
| 9 | Análises e gargalos (Fase 10) |
| 10 | Relatório (Fase 11). **Marco M4: entrega** |

## Referências

- Bergmann, P. et al. *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR, 2019.
- Documentação do [OpenCV](https://docs.opencv.org/), [scikit-image](https://scikit-image.org/docs/stable/) e [scikit-learn](https://scikit-learn.org/stable/).
