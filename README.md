# Detecção de Defeitos Industriais com PDI Clássico (MVTec AD · `bottle`)

Projeto 4 da disciplina de Processamento Digital de Imagens (DC/UFSCar, Prof. Dr. Ricardo J. Ferrari).
O objetivo é localizar, descrever e classificar defeitos em garrafas do dataset **MVTec AD** usando apenas
PDI clássico e atributos interpretáveis. Nenhuma etapa usa rede neural.

| Integrante | RA |
|---|---|
| Danilo Djovano Selli Junior | 842868 |
| Vitor Lima dos Santos | 845564 |

## Pergunta do projeto

> É possível localizar e classificar defeitos na categoria `bottle` do MVTec AD usando apenas PDI clássico e atributos interpretáveis?

O sistema tem duas saídas, avaliadas separadamente:

- **Localização:** máscara binária do defeito, comparada com a máscara ground truth (Dice, IoU, precisão e recall pixel a pixel).
- **Decisão:** normal × defeituosa (e, como extensão, o tipo de defeito), avaliada por F1, balanced accuracy e ROC-AUC.

O projeto é bem-sucedido quando responde à pergunta com números e aponta, com testes objetivos, qual etapa do
pipeline mais limita o desempenho. Um resultado negativo bem explicado também conta como resposta.

**Regra de método (vale para todas as etapas):** antes de testar alternativas, definimos (1) o objetivo da etapa,
(2) a hipótese sobre qual técnica deve funcionar e por quê e (3) a métrica que decide. Só então as alternativas
são comparadas, uma mudança por vez, contra um baseline fixo.

## Dataset

- **Fonte:** [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), categoria `bottle` (900×900 RGB, garrafas vistas de cima sobre fundo branco)
- **Licença:** CC BY-NC-SA 4.0 (uso não comercial)

| Pasta | Imagens |
|---|---|
| `train/good` | 209 |
| `test/good` | 20 |
| `test/broken_large` | 20 |
| `test/broken_small` | 22 |
| `test/contamination` | 21 |
| **Total** | **292** (63 defeituosas, todas com máscara) |

As contagens são conferidas automaticamente por `scripts/download_data.py` e `scripts/check_env.py`.

## O que os dados mostram

A exploração (Fase 1) veio antes de qualquer escolha de técnica e mudou partes do plano inicial.
Todas as medidas foram feitas a 256×256.

| Tipo | Área do defeito (% da imagem): mín · mediana · máx | Menor defeito (px) | Posição radial mediana (0 = centro, 1 = borda) | ΔL* vs. vizinhança (mediana) |
|---|---|---|---|---|
| `broken_large` | 4,1 · 11,9 · 27,4 | 2.680 | 0,39 (anel interno) | +11,5 (mais claro) |
| `broken_small` | 0,9 · 2,4 · 8,1 | 579 | 0,53 (anel do gargalo) | +24,4 (mais claro) |
| `contamination` | 0,6 · 7,3 · 16,5 | 377 | 0,10 (fundo da garrafa) | ±9,2 (sinal variado) |

1. **O ruído de sensor é praticamente nulo** (Immerkær, σ ≈ 0,5 nível de cinza). Por isso o denoising não serve para remover ruído (ver Fase 4).
2. **A variação entre garrafas boas está nos anéis, não no fundo.** O desvio por pixel entre as 209 boas fica entre 1 e 3 níveis no fundo e entre 9 e 17 nos anéis. A garrafa se desloca até ~5 px entre fotos. Desalinhamento e reflexo nos anéis são a principal fonte de falsos positivos.
3. **Quebras e contaminação têm assinaturas diferentes.** As quebras ficam nos anéis e são mais claras. As contaminações ficam no fundo e alteram a textura. A mudança de cor no plano (a, b) do Lab é ≤ 3,8 em mediana, o que enfraquece qualquer segmentação baseada só em cor.
4. **Os defeitos são regiões únicas de tamanho conhecido.** Há, em mediana, 1 componente por máscara, e o menor defeito tem **377 px** (diâmetro equivalente ≈ 22 px). Esse valor é o teto para σ dos filtros, para o elemento estruturante e para a área mínima de componente.
5. **As boas do teste são um pouco mais claras que as do treino**, provavelmente por terem sido fotografadas em outra sessão. Nelas aparece cerca do dobro de falsos positivos.

## Partição supervisionada própria

No MVTec AD as defeituosas só existem no teste, porque o protocolo oficial é não supervisionado. Seguindo a
especificação da disciplina, usamos uma partição própria em que **cada imagem tem um único papel**:

| Conjunto | Conteúdo | Papel |
|---|---|---|
| Referência | 120 boas do treino | Estimar média μ(x,y) e desvio σ(x,y) por pixel |
| Calibração | 30 boas do treino | Fixar limiares pela taxa de falso positivo em peças boas |
| Desenvolvimento | 3 defeituosas por tipo (9) | Escolher filtros, métodos e parâmetros; nunca entram nas métricas finais |
| Avaliação | 59 boas do treino + 20 boas do teste + 54 defeituosas (**133**) | Métricas de segmentação e validação cruzada dos classificadores |

Regras do protocolo:

- Validação cruzada estratificada por tipo (`good`, `broken_large`, `broken_small`, `contamination`), **5 folds × 5 sementes**, com resultados em média ± desvio.
- Hiperparâmetros escolhidos só nos folds de treino (`GridSearchCV` dentro de cada fold), com o `StandardScaler` dentro do `Pipeline`.
- As imagens de referência e calibração nunca entram na avaliação. Os testes automatizados (`tests/test_split.py`) verificam isso.
- A semente é única e registrada (`seed: 0` no `config.yaml`). A partição fica salva em `results/fase2/particao.csv`.
- Escolhas feitas com só 9 imagens de desenvolvimento usam a **AUROC de pixel** (~330 mil pixels, independente do limiar). Diferenças menores que **0,01** contam como empate, e no empate vence a opção mais simples.

> **Aviso:** os números deste projeto vêm de um experimento didático com partição própria e **não são comparáveis ao benchmark oficial do MVTec AD**.

## Pipeline

```
Imagem (256×256, cinza e Lab)
  → Alinhamento por translação (centroide da garrafa via Otsu) + ROI circular
  → Denoising (Gaussiano σ=2)
  → Mapa de anomalia: max( z de intensidade , z do gradiente Sobel )
  → Segmentação  — Método A: limiar calibrado nas boas  |  Método B: K-means intra-imagem
  → Morfologia (abertura, fechamento, remoção de componentes < A_min)
  → Atributos na máscara prevista (geometria, intensidade/cor, textura, contorno/HOG)
  → Classificação (SVM RBF | Random Forest)
```

Mapa de anomalia por pixel, calculado em relação ao modelo do normal:

```
z(x,y) = |I(x,y) − μ(x,y)| / (σ(x,y) + ε)
```

Dividir por σ dá automaticamente menos peso aos anéis, que variam naturalmente.

### Etapas, decisões e justificativas

| Fase | Etapa | Decisão | Por quê |
|---|---|---|---|
| 0 | Preparação | Redimensionar para 256×256 (`INTER_AREA` nas imagens, `INTER_NEAREST` nas máscaras); ROI circular | Iterar rápido sem perder o menor defeito; a ROI tira o fundo branco das métricas e dos limiares |
| 3 | Alinhamento e referência | Translação até o centro médio; μ e σ das 120 de referência; limiar = percentil 99,5 de z nas 30 de calibração | Deslocamentos de até 5 px geram falsos positivos nos anéis. O limiar é definido só com peças boas, como numa linha de produção. O Otsu por imagem marcou ~25% da garrafa boa como defeito |
| 4 | Denoising | **Gaussiano σ=2** (testados: nenhum, Gaussiano σ∈{1,2,3}, mediana 5×5, bilateral) | O objetivo é tolerar desalinhamento e reflexo, e não remover ruído. Mediana e bilateral preservam justamente as bordas dos anéis que geram erro. σ=3 empata com σ=2, e o menor é mais seguro para defeitos pequenos |
| 4 | Bordas | **Sobel:** z da diferença de gradiente em relação à referência, combinado por máximo. **Canny descartado** como detector | O Sobel realça o contorno das fraturas. Só 15 a 18% das bordas novas do Canny caem em defeitos |
| 4 | Normalização de brilho | Não adotada | Piora a AUROC e não reduz os falsos positivos nas boas do teste |
| 5 | Segmentação (Exp. 1) | A: limiar sobre o mapa z (variante: crescimento por histerese). B: K-means sobre L, a, b, desvio local 5×5 e raio normalizado | Pergunta: é preciso um modelo do normal, ou o defeito se destaca dentro da própria imagem? Como a cor sozinha muda pouco, o Método B usa também textura e posição |
| 6 | Morfologia (Exp. 2) | Abertura e fechamento com disco de raio 1 a 3; remoção de componentes com A_min < 377 px | A abertura tira arcos finos dos anéis, o fechamento une fragmentos das quebras. A_min vem da distribuição de componentes espúrios na calibração |
| 7 | Métricas de segmentação | Dice, IoU, precisão e recall implementados à mão em NumPy, dentro da ROI, por imagem e agregados | A média por imagem e a agregada respondem a perguntas diferentes (a agregada é dominada por `broken_large`). Nas boas, mede-se a fração de pixels FP e os componentes espúrios |
| 8 | Atributos | Calculados sobre a **máscara prevista**, nunca sobre a GT (exceto no teste do oráculo) | Usar a GT esconderia os erros da segmentação |
| 9 | Classificação (Exp. 3 e 4) | SVM RBF × Random Forest, `class_weight="balanced"`; regra trivial "máscara não vazia" como referência | Mede quanto o classificador acrescenta sobre a segmentação sozinha |

### Atributos (Fase 8)

| Grupo | Atributos | O que deve separar |
|---|---|---|
| Geometria | Área, perímetro, extensão, compacidade P²/A, solidez, excentricidade, **posição radial do centroide**, orientação relativa ao raio | Normal × defeito (área); quebra × contaminação (raio) |
| Intensidade/cor | Média e desvio de L, a, b; contraste região × anel externo; histograma de 8 faixas | Quebra × contaminação |
| Textura | LBP uniforme (P=8, R=1); GLCM (contraste, homogeneidade, energia, correlação) | Contaminação × normal |
| Contorno/HOG | Curvatura, concavidades, momentos de Hu, HOG reduzido (região 32×32, células 16×16) | `broken_large` × `broken_small` |

Com máscara vazia, a geometria é 0, o indicador `mascara_vazia = 1` é ativado e intensidade e textura são medidas numa janela em
torno do pixel de maior z. Com vários componentes, a geometria e o contorno vêm do maior componente, e o número de
componentes e a área total entram como atributos extras.

Busca de hiperparâmetros: SVM com C ∈ {0,1; 1; 10; 100} e gamma ∈ {scale; 0,01; 0,1}; RF com profundidade ∈ {None, 5, 10} e mínimo por folha ∈ {1, 3}.

## Experimentos

| # | Experimento | Pergunta |
|---|---|---|
| 1 | Método A × Método B | O modelo do normal é necessário para localizar o defeito? |
| 2 | Efeito da morfologia | Quanto cada operação reduz os falsos positivos e quanto custa em recall? (curva precisão × recall, escolha no joelho) |
| 3 | Grupos de atributos | Geometria × textura × todos (extensão: atributos locais × globais) |
| 4 | Classificadores | SVM × Random Forest na mesma partição |
| 5 | Resultados por tipo | Qual tipo é mais difícil, e o motivo é tamanho, contraste ou posição? |

Os Experimentos 3 e 4 rodam juntos: linhas = grupo de atributos, colunas = classificador × métrica.
A classificação em 4 classes (good + 3 tipos, F1 macro e matriz de confusão) é extensão. Com 18 a 19 exemplos por tipo, a variância é alta.

### Análise de gargalos

| Teste | Como | O que conclui |
|---|---|---|
| Oráculo de segmentação | Classificar com atributos calculados na máscara GT | Se o F1 sobe muito, o gargalo é a segmentação |
| Rastreio de falsos negativos | O defeito foi perdido na segmentação (Dice ≈ 0) ou foi segmentado e mal classificado? | Separa erro da segmentação de erro do classificador |
| Ablação por etapa | Retirar alinhamento, denoising, bordas e morfologia, um por vez | Tabela etapa → ΔDice, ΔF1 |
| Troca de classificador | SVM × RF com os mesmos atributos | Se empatam, o classificador não é o gargalo |

Inclui também uma galeria de 3 a 5 falsos positivos e falsos negativos (imagem, mapa z, máscara bruta, máscara final e GT), com a causa provável anotada.

## Estado atual

As Fases 0 a 4 estão concluídas: dados, exploração, partição, referência/baseline e denoising/bordas.
Resultados no conjunto de desenvolvimento:

| | Baseline (Fase 3) | + Gaussiano σ=2 | + Sobel (máximo) |
|---|---|---|---|
| AUROC de pixel (defeito × normal) | 0,83 | 0,89 | 0,90 |
| Dice | 0,52 | 0,64 | 0,63 |
| Componentes espúrios por garrafa boa (calibração) | 16,9 | 4,1 | 11,8 |

- **Sobel adotado com ressalva:** o ganho de AUROC (+0,013) fica perto da margem de empate, e os componentes espúrios quase triplicam. A Fase 6 decide: se a morfologia não limpar esses componentes, o pipeline volta ao mapa só de intensidade.
- **Maior risco:** a parte escura das quebras grandes parece garrafa normal, e nenhum mapa a captura até agora. No baseline, o Dice em `broken_large` é 0,27, contra 0,57 a 0,71 em `contamination`. Se a histerese e a morfologia não resolverem, isso vira uma limitação documentada da abordagem clássica.

**Próximo:** Fase 5 (Método B e histerese, Experimento 1) e Fase 6 (morfologia, Experimento 2).

## Estrutura do repositório

O repositório cresce junto com o projeto: um módulo, uma pasta ou uma dependência só entra quando a fase que o usa começa.

```
industrial-defect-detection/
├── config.yaml              parâmetros de todas as fases + seção `pipeline` com as escolhas feitas
├── requirements.txt
├── pyproject.toml           permite `pip install -e .`
├── data/                    dataset (baixado localmente, fora do Git)
├── notebooks/
│   ├── 00_verificacao_ambiente.ipynb
│   └── 01_exploracao.ipynb  interpretação da Fase 1
├── scripts/
│   ├── download_data.py     baixa e confere o dataset
│   ├── check_env.py         verifica se o ambiente está pronto
│   └── run_fase{1..4}.py    um script por fase; grava em results/faseN/
├── src/pdi_defeitos/
│   ├── config.py            leitura do config.yaml
│   ├── io.py                listagem do dataset, imagens e máscaras
│   ├── stats.py             medidas da exploração (ruído, disco da garrafa, geometria dos defeitos)
│   ├── split.py             partição supervisionada e folds
│   ├── reference.py         alinhamento, ROI, modelo de referência, Otsu manual
│   ├── preprocess.py        filtros, Sobel por convolução em NumPy, bordas novas do Canny
│   ├── metrics.py           Dice, IoU, precisão, recall, componentes
│   ├── pipeline.py          carregamento alinhado e protocolo único de avaliação
│   └── viz.py               galerias e painéis de figuras
├── results/faseN/           CSVs e figuras de cada fase
└── tests/                   testes automáticos (rodam sem o dataset)
```

Os parâmetros ficam no `config.yaml`. Cada fase registra na seção `pipeline` o que decidiu (com a justificativa em
comentário), e as fases seguintes leem dali.

## Configurar o ambiente (uma vez por computador)

### 1. Instalar os programas

| Programa | Onde | Observação |
| --- | --- | --- |
| Git | https://git-scm.com/downloads | No Windows, aceite as opções padrão do instalador. |
| Python 3.11 ou 3.12 | https://www.python.org/downloads/ | No Windows, **marque "Add python.exe to PATH"** na primeira tela. |
| VS Code | https://code.visualstudio.com/ | Ao abrir o repositório ele sugere as extensões Python e Jupyter: aceite. |

### 2. Clonar e criar o ambiente virtual

```
git clone https://github.com/vtorlima/industrial-defect-detection.git
cd industrial-defect-detection
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Se aparecer erro de "execução de scripts desabilitada", rode uma vez
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e tente de novo.

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Com o ambiente ativo, o terminal mostra `(.venv)` no começo da linha. **Ative sempre antes de trabalhar.**

### 3. Instalar as dependências

```
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
nbstripout --install
```

- `pip install -e .` instala o pacote `pdi_defeitos` em modo editável: mudanças em `src/` valem na hora.
- `nbstripout --install` faz o Git ignorar as saídas dos notebooks, o que evita conflitos de merge.

### 4. Baixar o dataset (~150 MB)

```
python scripts/download_data.py
```

Se o download falhar, baixe `bottle` manualmente na
[página do MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads) e rode:

```
python scripts/download_data.py --archive CAMINHO/PARA/bottle.tar.xz
```

Resultado: `data/raw/mvtec/bottle/{train,test,ground_truth}/`.

### 5. Verificar

```
python scripts/check_env.py     # deve terminar com TUDO OK
pytest                          # todos os testes passando
```

## Reproduzir os resultados

Rode da raiz, com o ambiente ativo e nesta ordem, porque cada fase lê o que a anterior gravou:

```
python scripts/run_fase1.py     # exploração      → results/fase1/
python scripts/run_fase2.py     # partição e folds → results/fase2/particao.csv, folds.csv
python scripts/run_fase3.py     # referência e baseline → results/fase3/
python scripts/run_fase4.py     # denoising e bordas    → results/fase4/
```

## Como trabalhamos com o Git

Regra geral: **`main` sempre funciona**. Cada fase é desenvolvida numa branch (`faseN-nome`) e entra na `main` por Pull Request, revisado pelo outro integrante.

```
git checkout main && git pull
git checkout -b fase5-segmentacao
# ... trabalhar ...
git add . && git commit -m "Fase 5: K-means intra-imagem"
git push -u origin fase5-segmentacao
```

Combinados:

- Código reaproveitável vai para `src/pdi_defeitos/`. Notebooks e scripts só chamam funções e mostram resultados.
- Parâmetros ficam no `config.yaml`, nunca espalhados no código.
- Nunca suba o dataset (o `.gitignore` já bloqueia).
- Rode `pytest` antes de abrir um Pull Request.

## Problemas comuns

| Sintoma | Solução |
| --- | --- |
| `python` não é reconhecido (Windows) | Reinstale o Python marcando "Add to PATH", ou use `py` no lugar de `python`. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ModuleNotFoundError: pdi_defeitos` | Ambiente não ativado ou faltou `pip install -e .` |
| Notebook não acha as bibliotecas | No VS Code, selecione o kernel `.venv` |
| `cv2.imread` devolve `None` | Caminho com acento no Windows; use `pdi_defeitos.io.load_image`, que já trata isso |

## Cronograma

Semanas relativas; o prazo final da disciplina ainda precisa ser confirmado.

| Semana | Fases | Marco | Status |
|---|---|---|---|
| 1 | 0–2: preparação, exploração, partição | Partição salva com semente | ✅ |
| 2 | 3 + métricas de segmentação | **M0:** baseline com Dice e FP medidos | ✅ |
| 3 | Pipeline mínimo de ponta a ponta (geometria + RF) | **M1:** primeira tabela de F1, mesmo ruim | ⏳ |
| 4 | 4: denoising e bordas | Filtro e variante de bordas escolhidos com critério | ✅ |
| 5 | 5–6: Método B e morfologia | **M2:** Experimentos 1 e 2 | ⏳ |
| 6–7 | 8–9: atributos completos e os dois classificadores | **M3:** Experimentos 3 e 4 | — |
| 8 | 10: resultados por tipo, galeria, testes de gargalo | Experimento 5 e análise de gargalos | — |
| 9–10 | 11: relatório e folga | **M4:** relatório revisado | — |

Se faltar tempo, cortamos nesta ordem: HOG, crescimento de regiões, classificação por tipo, extensão global × local.

## Referências

- Bergmann, P. et al. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR, 2019.
- Bergmann, P. et al. *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* IJCV, 2021.
- Immerkær, J. *Fast Noise Variance Estimation.* Computer Vision and Image Understanding, 1996.
- Documentação do [OpenCV](https://docs.opencv.org/), [scikit-image](https://scikit-image.org/docs/stable/) e [scikit-learn](https://scikit-learn.org/stable/).
