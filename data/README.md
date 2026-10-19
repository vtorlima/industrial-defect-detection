# Dados

Esta pasta não vai para o GitHub (está no `.gitignore`). Cada pessoa baixa os dados no próprio computador com:

```
python scripts/download_data.py
```

Depois do download a estrutura fica assim:

```
data/raw/mvtec/bottle/
├── train/good/                 209 imagens normais
├── test/good/                  20 imagens normais
├── test/broken_large/          20 defeituosas
├── test/broken_small/          22 defeituosas
├── test/contamination/         21 defeituosas
└── ground_truth/<tipo>/        uma máscara *_mask.png por imagem defeituosa
```

Fonte: [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), licença CC BY-NC-SA 4.0 (uso não comercial).
Não suba imagens do dataset para o repositório.
