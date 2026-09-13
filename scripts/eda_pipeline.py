import pandas as pd
import numpy as np

# PIPELINE DE ENGENHARIA DE DADOS - PROJETO SEMANTIX

# 1. Ingestão da Fonte de Dados Pública
url_data = 'https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv'
df_raw = pd.read_csv(url_data)

# 2. Seleção de Variáveis Críticas de Risco e Negócio
selected_columns = [
    'Duration', 'Amount', 'Age',
    'InstallmentRatePercentage', 'Housing', 'Class'
]
df_processed = df_raw[selected_columns].copy()

# 3. Padronização e Tradução dos Nomes de Colunas
df_processed.columns = [
    'Duracao_Meses', 'Valor_Emprestimo', 'Idade',
    'Taxa_Parcela_Renda', 'Moradia', 'Status_Original'
]

# 4. Engenharia de Recursos (Feature Engineering)
# 4.1 Binarização do Target: 1 para Inadimplente, 0 para Adimplente
df_processed['Inadimplente'] = df_processed['Status_Original'].apply(lambda x: 1 if x == 2 else 0)

# 4.2 Categorização Histórica por Faixa Etária
bins_idade = [18, 25, 35, 50, 100]
labels_idade = ['18-25', '26-35', '36-50', '50+']
df_processed['Faixa_Etaria'] = pd.cut(df_processed['Idade'], bins=bins_idade, labels=labels_idade)

# 4.3 Estimação do Valor da Parcela Mensal
df_processed['Parcela_Estimada_Mes'] = (
    df_processed['Valor_Emprestimo'] / df_processed['Duracao_Meses']
).round(2)

# 4.4 Categorização do Prazo Contratual
df_processed['Categoria_Prazo'] = pd.cut(
    df_processed['Duracao_Meses'],
    bins=[0, 12, 24, 36, 100],
    labels=['Curto Prazo (<=12m)', 'Médio Prazo (13-24m)', 'Longo Prazo (25-36m)', 'Super Longo (>36m)']
)

# 5. Exportação da Base Tratada
df_processed.to_csv('dados_credito_processados.csv', index=False)
print("Pipeline de EDA concluído com sucesso!")
