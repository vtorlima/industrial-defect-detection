"""Pipeline clássico de PDI para detecção e classificação de defeitos (MVTec AD).

Cada etapa do pipeline vira um módulo deste pacote quando a fase correspondente
começa. Módulos atuais:

  config  leitura do config.yaml
  io      listagem do dataset, imagens e máscaras
"""

__version__ = "0.1.0"
