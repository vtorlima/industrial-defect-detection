"""Pipeline clássico de PDI para detecção e classificação de defeitos (MVTec AD).

Cada etapa do pipeline vira um módulo deste pacote quando a fase correspondente
começa. Módulos atuais:

  config  leitura do config.yaml
  io      listagem do dataset, imagens e máscaras
  stats   medidas da exploração (ruído, posição, tamanho e contraste dos defeitos)
  viz     galerias e mapas de variabilidade
  split   partição supervisionada própria e folds da validação cruzada
  metrics Dice, IoU, precisão e recall pixel a pixel; falsos positivos nas boas
  reference alinhamento, média e desvio por pixel, mapa de anomalia z, Otsu
"""

__version__ = "0.1.0"
