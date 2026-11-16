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

**Regra de método (vale para todas as fases):** antes de testar alternativas, definimos (1) o objetivo da etapa,
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

## O que os dados mostram (Fase 1)

A exploração veio antes de qualquer escolha de técnica. Cada medida respondeu a uma pergunta feita antes de olhar o
resultado. Os números são a 256×256, exceto o ruído, que foi medido a 900×900.

| Tipo | n | Menor área (px) | Área mediana (px) | Posição radial mediana (0 = centro, 1 = borda) | ΔL vs. vizinhança (mediana) | Δab (mediana) |
|---|---|---|---|---|---|---|
| `broken_large` | 20 | 2.682 | 7.781 | 0,44 (anel interno) | +10,9 | 1,4 |
| `broken_small` | 22 | 582 | 1.597 | 0,60 (anel do gargalo) | +25,6 | 3,2 |
| `contamination` | 21 | 378 | 4.754 | 0,11 (fundo) | +2,0 | 3,8 |

ΔL é o brilho do defeito menos o de um anel de 6 px ao redor dele. Δab é a mudança de cor no plano (a, b) do Lab, sem contar o brilho.

1. **O ruído de sensor é desprezível:** σ ≈ 0,6 nível de cinza (0,58 a 0,72, estimador de Immerkær). O denoising serve para tolerar desalinhamento e reflexo, não para tirar ruído.
2. **A garrafa quase não se desloca, mas o pouco que se desloca já gera falsos positivos:** o centro tem desvio de 0,6 px e amplitude de ~3,4 px. Nas bordas nítidas dos anéis, isso basta para gerar diferenças.
3. **A garrafa boa varia de 5 a 10 vezes mais nos anéis que no fundo:** o desvio mediano entre boas é de 1,4 a 2,9 níveis no fundo e de 9,1 a 16,7 nos anéis, e passa de 50 nos pontos de reflexo.
4. **Cada tipo de defeito tem uma assinatura própria:** as quebras ficam nos anéis e são mais claras, e as contaminações ficam no fundo e mudam pouco o brilho. A cor muda pouco em todos os tipos, e a posição radial quase sozinha separa os três (0,11 × 0,44 × 0,60).
5. **O menor defeito tem 378 px** (diâmetro equivalente ≈ 22 px). Esse valor é o teto para σ dos filtros, para o elemento estruturante e para a área mínima de componente.

Hipóteses registradas na Fase 1, a confirmar ou refutar no relatório:

1. `contamination` será o tipo mais fácil de localizar, porque fica na região estável da garrafa.
2. `broken_small` será o mais difícil de detectar, por ser pequeno e estar no anel mais variável.
3. A geometria (área e posição) basta para separar normal × defeituosa e quebra × contaminação. A textura só será necessária nas contaminações sutis.

## Partição supervisionada própria (Fase 2)

No MVTec AD as defeituosas só existem no teste, porque o protocolo oficial é não supervisionado. Seguindo a
especificação da disciplina, usamos uma partição própria com semente 0, em que **cada imagem tem um único papel**:

| Papel | Boas | broken_large | broken_small | contamination | Total | Para que serve |
|---|---|---|---|---|---|---|
| Referência | 120 | 0 | 0 | 0 | 120 | Estimar média μ(x,y) e desvio σ(x,y) por pixel |
| Calibração | 30 | 0 | 0 | 0 | 30 | Fixar limiares só com peças boas |
| Desenvolvimento | 0 | 3 | 3 | 3 | 9 | Escolher filtros, métodos e parâmetros; fora das métricas finais |
| Avaliação | 79 (59 treino + 20 teste) | 17 | 19 | 18 | 133 | Todas as métricas e a validação cruzada |

Regras do protocolo:

- Validação cruzada estratificada por tipo, **5 folds × 5 sementes** (25 rodadas). Cada fold de teste tem 3 ou 4 imagens de cada defeito e 15 ou 16 boas. Os resultados são reportados em média ± desvio, e diferenças menores que o desvio não contam como ganho.
- Hiperparâmetros escolhidos só nos folds de treino (`GridSearchCV` dentro de cada fold), com o `StandardScaler` dentro do `Pipeline`.
- `check_partition` recusa a partição se alguma imagem aparecer em dois papéis ou se uma defeituosa cair na referência. Um teste provoca esse vazamento de propósito para confirmar que ele é detectado.
- As fases seguintes leem `results/fase2/particao.csv` e nunca sorteiam de novo. Rodar o script outra vez gera arquivos idênticos.
- O desenvolvimento (3 imagens por tipo) só mostra direção. As decisões usam a **AUROC de pixel** (~330 mil pixels, independente do limiar). Diferenças menores que **0,01** contam como empate, e no empate vence a opção mais simples.
- As boas do teste são ≈ 2 níveis de cinza mais claras que as do treino, provavelmente de outra sessão de fotos. Por isso os falsos positivos são reportados separadamente para boas do treino e do teste.

> **Aviso:** os números deste projeto vêm de um experimento didático com partição própria e **não são comparáveis ao benchmark oficial do MVTec AD**.

## Pipeline

```
Imagem (256×256, cinza e Lab)
  → Gaussiano σ=2
  → Alinhamento por translação (centroide da garrafa via Otsu) + ROI (disco com 95% do raio)
  → Mapa de anomalia: z de intensidade  |  max(z intensidade, z gradiente Sobel)
  → Segmentação — Método A: limiar calibrado (+ histerese)  |  Método B: K-means intra-imagem
  → Morfologia (abertura, fechamento, remoção de componentes < A_min)
  → Atributos na máscara prevista (geometria, intensidade/cor, textura, contorno/HOG)
  → Classificação (SVM RBF | Random Forest)
```

Mapa de anomalia por pixel, calculado em relação ao modelo do normal:

```
z(x,y) = |I(x,y) − μ(x,y)| / (σ(x,y) + ε),   ε = 2
```

Dividir por σ dá automaticamente menos peso aos anéis, que variam naturalmente. O ε = 2 evita que pixels quase
constantes gerem z enorme. Os dois mapas de anomalia (só intensidade e máximo com Sobel) seguem até a Fase 6, e a
morfologia decide entre eles.

### Fases, decisões e justificativas

| Fase | Etapa | Decisão | Por quê |
|---|---|---|---|
| 0 | Estrutura | Pacote `pdi_defeitos`, `config.yaml` central, testes automáticos | Uma fonte única de parâmetros e resultados reproduzíveis nas duas máquinas |
| 1 | Exploração | Medir ruído, posição, variabilidade e defeitos antes de escolher técnicas | Cada escolha seguinte sai com hipótese e faixa permitida |
| 2 | Partição | Partição própria com semente fixa e validação cruzada 5×5 | Sem protocolo fixo, nenhuma comparação é confiável |
| 3 | Referência e baseline | Alinhar por translação até o centro médio (127,77; 127,96); μ e σ das 120 de referência; limiar = percentil 99,5 de z nas 30 de calibração (2,63) | O alinhamento reduz em 10% a variação nos anéis e melhora Dice, AUROC e fragmentação. O **Otsu por imagem foi descartado**: ele marca ~25% da garrafa boa, porque sem um modelo do normal o limiar não sabe quando não há defeito |
| 3 (7) | Métricas de segmentação | Dice, IoU, precisão e recall feitos à mão em NumPy, só dentro da ROI, por imagem e agregados | A média por imagem e a agregada respondem perguntas diferentes (em `contamination`, 0,57 × 0,78). Nas boas medem-se pixels FP e componentes espúrios |
| 4 | Denoising | **Gaussiano σ=2** (testados: nenhum, Gaussiano σ∈{1,2,3}, mediana 5×5, bilateral) | Mediana e bilateral preservam as bordas dos anéis, que são a fonte de erro. σ=3 empata (diferença de 0,005), e σ=2 tem menos risco para defeitos pequenos |
| 4 | Normalização de brilho | **Não usar** | Piora a AUROC (−0,014) e não reduz os FP extras das boas do teste. A causa é local, não o brilho médio |
| 4 | Bordas | **Sobel:** z da magnitude do gradiente, combinado pelo máximo com o z de intensidade. **Canny descartado** | Pela regra pré-definida (ganho > 0,01 na média das quebras sem perder > 0,01 em contamination), o máximo foi aprovado: +0,024 em `broken_large`. Custa ~3× mais componentes espúrios. No Canny, 82 a 85% das bordas novas ficam fora dos defeitos |
| 5 | Segmentação (Exp. 1) | A: limiar calibrado, com **crescimento por histerese como prioridade**. B: K-means sobre L, a, b, desvio local 5×5 e raio normalizado | Pergunta: é preciso um modelo do normal, ou o defeito se destaca dentro da própria imagem? Como o recall está baixo (0,53) e a precisão alta, a histerese deve recuperar a parte do defeito que o limiar perde |
| 6 | Morfologia (Exp. 2) | Abertura e fechamento com disco de raio 1 a 3; remoção de componentes com A_min < 378 px | A abertura tira arcos finos dos anéis, o fechamento une fragmentos das quebras. Fica o mapa com menos FP e recall igual ou maior depois da limpeza |
| 8 | Atributos | Calculados sobre a **máscara prevista**, nunca sobre a GT (exceto no teste do oráculo) | Usar a GT esconderia os erros da segmentação |
| 9 | Classificação (Exp. 3 e 4) | SVM RBF × Random Forest, `class_weight="balanced"`; regra trivial "máscara não vazia" como referência | Mede quanto o classificador acrescenta à segmentação. A regra trivial só faz sentido depois da morfologia, porque no baseline nenhuma imagem sai com máscara vazia |

### Atributos (Fase 8)

| Grupo | Atributos | O que deve separar |
|---|---|---|
| Geometria | Área, perímetro, extensão, compacidade P²/A, solidez, excentricidade, **posição radial do centroide**, orientação relativa ao raio | Normal × defeito (área); quebra × contaminação (raio) |
| Intensidade/cor | Média e desvio de L, a, b; contraste região × anel externo; histograma de 8 faixas; percentil 99 de z | Quebra × contaminação; normal × defeito |
| Textura | LBP uniforme (P=8, R=1); GLCM (contraste, homogeneidade, energia, correlação) | Contaminação × normal |
| Contorno/HOG | Curvatura, concavidades, momentos de Hu, HOG reduzido (região 32×32, células 16×16) | `broken_large` × `broken_small` |

Com máscara vazia, a geometria é 0, o indicador `mascara_vazia = 1` é ativado e intensidade e textura são medidas numa
janela em torno do pixel de maior z. Com vários componentes, a geometria e o contorno vêm do maior componente, e o
número de componentes e a área total entram como atributos extras.

## Experimentos

| # | Experimento | Pergunta |
|---|---|---|
| 1 | Método A × Método B | O modelo do normal é necessário para localizar o defeito? |
| 2 | Efeito da morfologia | Quanto cada operação reduz os falsos positivos e quanto custa em recall? (curva precisão × recall, escolha no joelho) |
| 3 | Grupos de atributos | Geometria × textura × todos (extensão: atributos locais × globais) |
| 4 | Classificadores | SVM × Random Forest na mesma partição |
| 5 | Resultados por tipo | Qual tipo é mais difícil, e o motivo é tamanho, contraste ou posição? |

A classificação em 4 classes (good + 3 tipos, F1 macro e matriz de confusão) é extensão. Com 17 a 19 exemplos por tipo na avaliação, a variância é alta.

### Análise de gargalos (Fase 10)

| Teste | Como | O que conclui |
|---|---|---|
| Oráculo de segmentação | Classificar com atributos calculados na máscara GT | Se o F1 sobe muito, o gargalo é a segmentação |
| Rastreio de falsos negativos | O defeito foi perdido na segmentação (Dice ≈ 0) ou foi segmentado e mal classificado? | Separa erro da segmentação de erro do classificador |
| Ablação por etapa | Retirar alinhamento, denoising, bordas e morfologia, um por vez | Tabela etapa → ΔDice, ΔF1 |
| Troca de classificador | SVM × RF com os mesmos atributos | Se empatam, o classificador não é o gargalo |

A fase inclui também uma galeria de falsos positivos e falsos negativos, com a causa provável anotada. Ela deve
explicar por que as boas do teste geram o dobro de falsos positivos.

## Estado atual

As Fases 0 a 4 estão concluídas. Resultados no conjunto de desenvolvimento:

| | Baseline (Fase 3) | + Gaussiano σ=2 | + Sobel (máximo) |
|---|---|---|---|
| AUROC de pixel | 0,83 | 0,89 | 0,90 |
| Dice | 0,52 | 0,64 | 0,63 |
| Recall | 0,40 | 0,53 | 0,53 |
| Componentes espúrios por garrafa boa (calibração) | 16,9 | 4,1 | 11,8 |

Baseline por tipo, no desenvolvimento: Dice de 0,28 em `broken_large`, 0,58 em `broken_small` e 0,71 em `contamination`.
A precisão é alta (0,76 a 0,93) e o recall é baixo (0,18 a 0,59), com 30 a 40 componentes por imagem.

- **O Sobel está condicionado.** Se a morfologia (Fase 6) não limpar os componentes extras, o pipeline volta ao mapa só de intensidade.
- **Maior risco:** a parte escura das quebras grandes tem intensidade e textura parecidas com as da garrafa normal e não aparece em nenhum mapa. Se o fechamento morfológico e a histerese não resolverem, isso vira uma limitação documentada da abordagem por referência.
- **Reprodutibilidade:** os números oficiais são os commitados em `results/`. Execuções em ambientes diferentes variam na terceira casa decimal por causa da interpolação do OpenCV. Por isso Dice, IoU e AUROC são reportados com duas casas.

**Próximo:** Fase 5 (Método A com histerese × Método B, Experimento 1) e Fase 6 (morfologia, Experimento 2).

## Estrutura do repositório

```
industrial-defect-detection/
├── config.yaml              parâmetros de todas as fases + seção `pipeline` com as escolhas feitas
├── requirements.txt
├── requirements-lock.txt    versões exatas testadas, para reproduzir os números de results/
├── pyproject.toml           permite `pip install -e .`
├── data/                    dataset (baixado localmente, fora do Git)
├── notebooks/
│   ├── 00_verificacao_ambiente.ipynb
│   └── 01_exploracao.ipynb  interpretação da Fase 1
├── scripts/
│   ├── download_data.py     extrai e confere o dataset
│   ├── check_env.py         verifica se o ambiente está pronto
│   ├── run_fase{1..4}.py    um script por fase; grava em results/faseN/
│   └── figuras_fase4.py     figuras da apresentação, só a partir dos CSVs de results/
├── src/pdi_defeitos/
│   ├── config.py            leitura do config.yaml
│   ├── io.py                listagem do dataset, imagens e máscaras
│   ├── stats.py             medidas da exploração (ruído, disco da garrafa, geometria dos defeitos)
│   ├── split.py             partição supervisionada, folds e verificação contra vazamento
│   ├── reference.py         alinhamento, ROI, modelo de referência, Otsu feito à mão
│   ├── preprocess.py        filtros, Sobel por convolução em NumPy, bordas novas do Canny
│   ├── metrics.py           Dice, IoU, precisão, recall, componentes
│   ├── pipeline.py          carregamento alinhado e protocolo único de avaliação (`evaluate_map`)
│   └── viz.py               galerias, painéis de figuras e gráficos de resultados
├── results/faseN/           CSVs e figuras de cada fase
└── tests/                   testes automáticos (rodam sem o dataset)
```

Os parâmetros ficam no `config.yaml`. Cada fase registra na seção `pipeline` o que decidiu, com a justificativa em
comentário, e as fases seguintes leem dali. Cada fase tem também um relatório próprio com a justificativa de cada escolha.

## Configurar o ambiente (uma vez por computador)

### 1. Instalar os programas

| Programa | Observação |
| --- | --- |
| Git | No Windows, aceite as opções padrão do instalador. |
| Python **3.11 ou 3.12** | No Windows, **marque "Add python.exe to PATH"**. No Fedora, o `python3` padrão pode ser mais novo: instale com `sudo dnf install python3.12` e use `python3.12` no passo 3. |
| VS Code | Ao abrir o repositório ele sugere as extensões Python e Jupyter: aceite. |

### 2. Clonar

```
git clone https://github.com/vtorlima/industrial-defect-detection.git
cd industrial-defect-detection
```

### 3. Criar e ativar o ambiente virtual

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Se aparecer erro de "execução de scripts desabilitada", rode uma vez
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` e tente de novo.

**macOS / Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Com o ambiente ativo, o terminal mostra `(.venv)` no começo da linha. **Ative sempre antes de trabalhar.**

### 4. Instalar as dependências

```
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
nbstripout --install
```

- Para reproduzir exatamente os números de `results/`, use `pip install -r requirements-lock.txt` no lugar de `pip install -r requirements.txt`: ele fixa as versões testadas das dependências.
- `pip install -e .` instala o pacote `pdi_defeitos` em modo editável: mudanças em `src/` valem na hora.
- `nbstripout --install` faz o Git ignorar as saídas dos notebooks. Confira com `nbstripout --status`. Se recriar o `.venv`, rode de novo, porque o filtro aponta para o Python do ambiente.

### 5. Obter o dataset (~150 MB)

O download exige um formulário na página oficial. Em
[mvtec.com › MVTec AD › Downloads](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads),
baixe só a categoria `bottle` e rode:

```
python scripts/download_data.py --archive CAMINHO/PARA/bottle.tar.xz
```

Resultado: `data/raw/mvtec/bottle/{train,test,ground_truth}/`. O script confere as contagens esperadas.

### 6. Verificar

```
python scripts/check_env.py     # deve terminar com TUDO OK
pytest                          # todos os testes passando
```

## Reproduzir os resultados

Rode da raiz, com o ambiente ativo e nesta ordem, porque cada fase lê o que a anterior gravou:

```
python scripts/run_fase1.py     # exploração            → results/fase1/
python scripts/run_fase2.py     # partição e folds      → results/fase2/
python scripts/run_fase3.py     # referência e baseline → results/fase3/
python scripts/run_fase4.py     # denoising e bordas    → results/fase4/
```

## Como trabalhamos com o Git

Regra geral: **`main` sempre funciona**. Cada fase é desenvolvida numa branch (`faseN-nome`) e entra na `main` por
Pull Request, revisado pelo outro integrante.

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
- Se um `.ipynb` aparecer como modificado sem você ter mexido, é o filtro do `nbstripout`: `git add --renormalize notebooks/` e commit.

## Problemas comuns

| Sintoma | Solução |
| --- | --- |
| `python` não é reconhecido (Windows) | Reinstale o Python marcando "Add to PATH", ou use `py` no lugar de `python`. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ModuleNotFoundError: pdi_defeitos` | Ambiente não ativado ou faltou `pip install -e .` |
| Notebook não acha as bibliotecas | No VS Code, selecione o kernel `.venv` |
| `cv2.imread` devolve `None` | Caminho com acento no Windows; use `pdi_defeitos.io.load_image`, que já trata isso |
| `download_data.py` dá HTTP 404 | O link direto mudou; baixe pelo formulário (passo 5) e use `--archive` |
| `git checkout` reclama de notebooks modificados | Filtro do `nbstripout`; veja o último item dos combinados acima |

## Cronograma

Semanas relativas; o prazo final da disciplina ainda precisa ser confirmado.

| Semana | Fases | Marco | Status |
|---|---|---|---|
| 1 | 0–2: estrutura, exploração, partição | Partição salva com semente | ✅ |
| 2 | 3: referência, baseline e métricas | **M0:** baseline com Dice e FP medidos | ✅ |
| 3 | Pipeline mínimo de ponta a ponta (geometria + RF) | **M1:** primeira tabela de F1, mesmo ruim | ⏳ |
| 4 | 4: denoising e bordas | Filtro e variante de bordas escolhidos com critério | ✅ |
| 5 | 5–6: segmentação e morfologia | **M2:** Experimentos 1 e 2 | ⏳ |
| 6–7 | 8–9: atributos completos e os dois classificadores | **M3:** Experimentos 3 e 4 | — |
| 8 | 10: resultados por tipo, galeria, testes de gargalo | Experimento 5 e análise de gargalos | — |
| 9–10 | 11: relatório e folga | **M4:** relatório revisado | — |

Se faltar tempo, cortamos nesta ordem: HOG, classificação por tipo, extensão global × local.

## Referências

- Bergmann, P. et al. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* CVPR, 2019.
- Bergmann, P. et al. *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* IJCV, 2021.
- Immerkær, J. *Fast Noise Variance Estimation.* Computer Vision and Image Understanding, 1996.
- Documentação do [OpenCV](https://docs.opencv.org/), [scikit-image](https://scikit-image.org/docs/stable/) e [scikit-learn](https://scikit-learn.org/stable/).