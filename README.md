# Bootcamp-Data-DIO-Python-Projeto-Final.2
Projeto sobre detecção de fraudes bancarias, utilizando conceitos e ferramentas aprendidas.

README sugerido
O README deve conter:

Problema
Explique que o objetivo é identificar transações fraudulentas em uma base altamente desbalanceada, em que a maioria absoluta das operações é legítima.

Dados
Informe:

Fonte do dataset.

Quantidade de registros.

Quantidade de fraudes.

Significado de Class.

Natureza anonimizada das variáveis PCA.

Preparação
Descreva:

Inspeção de valores ausentes.

Criação de LogAmount.

Separação estratificada.

Padronização.

Tratamento do desbalanceamento.

Ausência do CSV no repositório.

Modelos
Apresente uma tabela semelhante a esta, preenchida com os resultados reais do seu notebook:

Modelo	Limiar	Precisão fraude	Recall fraude	F1 fraude	ROC-AUC	PR-AUC
Regressão Logística	0,50	resultado	resultado	resultado	resultado	resultado
Random Forest	0,50	resultado	resultado	resultado	resultado	resultado
XGBoost	0,50	resultado	resultado	resultado	resultado	resultado
Modelo escolhido	escolhido	resultado	resultado	resultado	resultado	resultado
Não copie números de outro projeto. As métricas dependem da divisão dos dados, das variáveis, do random state, do tratamento aplicado e dos hiperparâmetros.

Limiar
Explique:

Qual limiar foi escolhido.

Qual critério foi usado.

Como o recall mudou.

Qual foi o impacto sobre precisão e falsos positivos.

SHAP
Descreva:

Quais variáveis apareceram com maior impacto global.

Se aumentaram ou reduziram a previsão de fraude.

Que a interpretação das variáveis PCA é limitada pela anonimização.

O que você mudou
Inclua alterações próprias, por exemplo:

Inclusão de LogAmount.

Comparação entre três algoritmos.

Avaliação com PR-AUC.

Ajuste do limiar.

Teste de undersampling e oversampling.

Explicação individual com SHAP.

Critério de seleção baseado em recall mínimo.

Checklist final
Antes de enviar o link:

O notebook executa do início ao fim com Restart Kernel and Run All.

Todas as saídas importantes estão salvas.

As curvas ROC e precisão-recall aparecem no notebook.

As métricas foram calculadas para a classe Fraude.

O dataset não está dentro do repositório.

O notebook carrega os dados pelo link ou documenta corretamente o download.

O README menciona o desbalanceamento.

O README contém a tabela real de resultados.

O README explica o limiar escolhido.

O README explica o SHAP.

Todos os arquivos mencionados existem.

O repositório é público.

O nome está em minúsculas, sem acentos e sem espaços.

O link entregue é o link do repositório, não o link direto do notebook.

O aspecto mais importante do projeto não é obter o maior número de acurácia, mas demonstrar que você reconheceu o desbalanceamento, escolheu métricas coerentes, ajustou o limiar conscientemente e conseguiu explicar as decisões do modelo.
