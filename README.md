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

## Estrutura do repositório

O repositório cresce junto com o projeto: um módulo, uma pasta ou uma dependência só entra quando a fase que o usa começa.
Estado atual (Fase 0):

```
industrial-defect-detection/
├── config.yaml            parâmetros centrais (categoria, caminhos, tamanho da imagem)
├── requirements.txt       bibliotecas usadas até agora
├── pyproject.toml         permite `pip install -e .`
├── data/                  dataset (baixado localmente, fora do Git)
├── notebooks/
│   └── 00_verificacao_ambiente.ipynb
├── scripts/
│   ├── download_data.py   baixa e confere o dataset
│   └── check_env.py       verifica se o ambiente está pronto
├── src/pdi_defeitos/
│   ├── config.py          leitura do config.yaml
│   └── io.py              listagem do dataset, imagens e máscaras
└── tests/                 testes automáticos (rodam sem o dataset)
```

## Tecnologias

Pilha prevista para o projeto inteiro. Cada biblioteca entra no `requirements.txt` quando a fase que a usa começa.

- Python 3.11 ou 3.12
- OpenCV
- scikit-image
- scikit-learn
- NumPy
- Matplotlib

## Configurar o ambiente (uma vez por computador)

### 1. Instalar os programas

| Programa | Onde | Observação |
| --- | --- | --- |
| Git | https://git-scm.com/downloads | No Windows, aceite as opções padrão do instalador. |
| Python 3.11 ou 3.12 | https://www.python.org/downloads/ | No Windows, **marque "Add python.exe to PATH"** na primeira tela. |
| VS Code | https://code.visualstudio.com/ | Ao abrir o repositório ele sugere as extensões Python e Jupyter: aceite. |

Confira no terminal (no Windows, PowerShell):

```
git --version
python --version
```

### 2. Clonar o repositório

```
cd C:\repos            # ou a pasta onde você guarda projetos
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
python3 -m venv .venv
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

- `pip install -e .` instala o pacote `pdi_defeitos` em modo editável: mudanças em `src/` valem na hora, em notebooks e scripts.
- `nbstripout --install` faz o Git ignorar as saídas dos notebooks, o que evita conflitos de merge entre nós dois.

### 5. Baixar o dataset (~150 MB)

```
python scripts/download_data.py
```

Se o download falhar (rede bloqueada ou link mudado), baixe `bottle` manualmente na
[página de downloads do MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads) e rode:

```
python scripts/download_data.py --archive CAMINHO\PARA\bottle.tar.xz
```

### 6. Verificar

```
python scripts/check_env.py
pytest
```

O primeiro deve terminar com **TUDO OK**; o segundo com todos os testes passando.
Por fim, abra `notebooks/00_verificacao_ambiente.ipynb` no VS Code, escolha o kernel **.venv** (canto superior direito) e rode tudo: deve aparecer uma garrafa de cada tipo com o defeito em vermelho.

## Como trabalhamos com o Git

Regra geral: **`main` sempre funciona**. Trabalho novo vai para uma branch e entra na `main` por Pull Request.

```
git checkout main
git pull                                   # 1. pegar o que o outro fez
git checkout -b fase1-exploracao           # 2. branch com nome da fase/tarefa
# ... trabalhar ...
git add .
git commit -m "Fase 1: galeria de imagens e áreas dos defeitos"
git push -u origin fase1-exploracao        # 3. enviar
```

Depois abra o Pull Request no GitHub; o outro revisa e faz o merge.

Combinados:

- Um notebook por pessoa por vez. Para editar o notebook do outro, combine antes.
- Código reaproveitável vai para `src/pdi_defeitos/`; notebooks só chamam funções e mostram resultados.
- Parâmetros ficam no `config.yaml` ou no topo do notebook, nunca espalhados no código.
- Nunca suba o dataset nem imagens geradas (o `.gitignore` já bloqueia).
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
- Bergmann, P. et al. *The MVTec Anomaly Detection Dataset: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection.* IJCV, 2021.
- Documentação do [OpenCV](https://docs.opencv.org/), [scikit-image](https://scikit-image.org/docs/stable/) e [scikit-learn](https://scikit-learn.org/stable/).
