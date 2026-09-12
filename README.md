# Goodreads — Web Scraping e Análise Exploratória de Dados

Projeto de coleta, tratamento e análise de dados da lista **Best Books Ever**, do Goodreads, desenvolvido em Python a partir de técnicas de **web scraping**.

O projeto percorre o processo completo de obtenção dos dados: extração das páginas da lista, estruturação e tratamento da base, construção de métricas analíticas, análise exploratória e desenvolvimento de um **relatório interativo para exploração dos livros coletados**.

## Sobre o projeto

Este trabalho surgiu no contexto da disciplina de **Laboratório em Estatística Aplicada**, do curso de Estatística da UFSCar.

A proposta inicial era aplicar técnicas de web scraping a uma fonte real de dados. A lista *Best Books Ever*, do Goodreads, apresentou uma oportunidade interessante de expandir o exercício para um projeto de análise de dados mais completo, explorando não apenas a coleta das informações, mas também questões como:

* quais livros aparecem com maior destaque na lista;
* quais autores possuem maior presença;
* quais obras concentram maior quantidade de avaliações;
* como diferenciar livros muito bem avaliados de livros bem avaliados por um público realmente grande;
* como o `score` da lista se relaciona com avaliação, popularidade e posição no ranking.

A partir disso, o projeto evoluiu para um pipeline de **Web Scraping + Análise Exploratória de Dados**, culminando em um relatório voltado à consulta e exploração dos resultados.

---

## Coleta dos dados

Os dados foram coletados diretamente das páginas da lista **Best Books Ever** utilizando requisições HTTP e parsing do HTML.

Foram percorridas as **25 primeiras páginas da lista**, com aproximadamente **100 livros por página**, resultando em uma base com cerca de **2.500 livros** antes das etapas de verificação e remoção de possíveis duplicidades.

Entre as informações coletadas estão:

| Variável             | Descrição                                            |
| -------------------- | ---------------------------------------------------- |
| `ranking`            | posição do livro na lista                            |
| `titulo`             | título da obra                                       |
| `autor`              | nome do autor                                        |
| `avaliacao_media`    | avaliação média no Goodreads                         |
| `numero_avaliacoes`  | quantidade de avaliações recebidas                   |
| `score`              | pontuação do livro na lista *Best Books Ever*        |
| `numero_votos_lista` | quantidade de votos recebidos na lista               |
| `url_livro`          | endereço da página do livro no Goodreads             |
| `url_autor`          | endereço da página do autor                          |
| `url_capa`           | endereço da imagem de capa                           |
| `goodreads_author`   | identificação de autores com selo *Goodreads Author* |

Além da extração, o processo inclui tentativas automáticas em caso de falha nas requisições, pausas entre acessos e salvamento de arquivos intermediários durante a coleta.

---

## Tratamento e preparação

Após a coleta, os dados passam por uma etapa de preparação para garantir consistência antes das análises.

Entre os principais tratamentos realizados estão:

* organização dos dados em um `DataFrame`;
* remoção de duplicidades utilizando o identificador do livro;
* conversão das variáveis quantitativas para formatos numéricos;
* análise de valores ausentes;
* ordenação dos registros pelo ranking da lista;
* criação de variáveis auxiliares para análise;
* criação de faixas de avaliação e popularidade;
* preparação de uma base específica para utilização no relatório.

O projeto inicialmente avaliou a possibilidade de acrescentar os gêneros dos livros. Essa informação, entretanto, não pôde ser obtida de maneira suficientemente consistente: as páginas individuais do Goodreads apresentaram limitações para a extração e fontes alternativas retornaram classificações pouco padronizadas.

Por esse motivo, a análise de gêneros foi retirada da versão final. A decisão prioriza a confiabilidade das informações efetivamente utilizadas no projeto.

---

## Nota ponderada

Um dos pontos centrais da análise foi perceber que **ordenar os livros apenas pela avaliação média pode produzir resultados pouco representativos**.

Um livro com nota muito alta, mas poucas avaliações, não possui necessariamente a mesma evidência de preferência coletiva que outro livro com nota semelhante e centenas de milhares ou milhões de avaliações.

Para lidar com esse problema, foi construída uma **média bayesiana ponderada**:

$$
WR = \frac{v}{v+m}R + \frac{m}{v+m}C
$$

em que:

* `R` representa a avaliação média do livro;
* `v` representa sua quantidade de avaliações;
* `C` representa a avaliação média global da base;
* `m` corresponde ao percentil 75 da quantidade de avaliações.

Essa abordagem reduz a influência de avaliações extremas sustentadas por poucas observações e permite construir um ranking que considera simultaneamente **qualidade percebida e volume de avaliações**.

---

## Análise exploratória

A análise foi organizada para observar os livros por diferentes perspectivas, evitando depender de uma única métrica de sucesso.

Foram avaliados, entre outros:

* distribuição das avaliações médias;
* livros mais avaliados;
* livros com maior nota ponderada;
* livros com maior `score` na lista;
* autores com mais livros no ranking;
* autores com maior volume acumulado de avaliações;
* relação entre avaliação e popularidade;
* relação entre ranking e `score`;
* quantidade de votos recebidos na lista;
* diferenças entre avaliação média e nota ponderada.

---

## Principais insights

### 1. Avaliação média, sozinha, não é suficiente para comparar os livros

Uma das principais conclusões do projeto foi a necessidade de considerar o **volume de avaliações junto à nota média**.

Livros com avaliações muito altas podem possuir bases de leitores bastante diferentes. A nota ponderada introduz esse contexto e produz uma comparação mais robusta entre as obras.

### 2. Popularidade e avaliação representam dimensões diferentes

A quantidade de avaliações apresenta grande variação entre os livros da lista. Algumas obras acumulam um volume de avaliações muito superior às demais, enquanto isso não significa necessariamente que possuam as maiores notas médias.

Por esse motivo, popularidade e avaliação foram tratadas separadamente e posteriormente combinadas na nota ponderada.

### 3. O `score` da lista não deve ser interpretado como uma avaliação do livro

A avaliação média representa a opinião dos usuários sobre a obra, enquanto o `score` e a quantidade de votos estão associados à participação do livro especificamente na lista *Best Books Ever*.

Comparar essas medidas permite distinguir **avaliação geral da obra** de **força dentro do ranking analisado**.

### 4. A presença de um autor pode ser analisada por mais de uma perspectiva

A análise por autor permite separar pelo menos dois conceitos:

* quantidade de livros presentes na lista;
* volume total de avaliações acumulado por suas obras.

Assim, um autor pode se destacar pela recorrência no ranking, pela popularidade de seus livros ou por ambos.

### 5. Rankings diferentes respondem a perguntas diferentes

O projeto trabalha com diferentes critérios de ordenação:

* ranking original do Goodreads;
* quantidade de avaliações;
* `score` da lista;
* avaliação média;
* nota ponderada.

Não existe, portanto, um único conceito de "melhor livro". Cada ranking representa uma dimensão diferente da preferência dos usuários.

---

## Relatório final

A etapa final do projeto transforma as análises em um **relatório interativo**, pensado para facilitar a exploração da base sem a necessidade de acessar diretamente os códigos ou notebooks.

O relatório reúne indicadores gerais, rankings e uma tabela detalhada dos livros.

Entre suas funcionalidades estão:

* consulta aos principais indicadores da base;
* visualização dos livros mais bem posicionados;
* ranking baseado na nota ponderada;
* comparação entre avaliação média e quantidade de avaliações;
* consulta aos autores com maior presença na lista;
* visualização dos livros com maior `score`;
* destaque visual das métricas de avaliação e nota ponderada;
* **pesquisa de livros pelo título**;
* consulta individual das informações de cada obra;
* **acesso direto à página original do livro no Goodreads a partir do relatório**.

A possibilidade de pesquisar uma obra e abrir sua página original conecta a análise ao dado de origem e transforma o relatório em uma ferramenta de exploração do catálogo, e não apenas em uma apresentação estática de gráficos.

---

## Pipeline do projeto

```text
Goodreads
    │
    ▼
Web Scraping
Requests + BeautifulSoup
    │
    ▼
Base bruta
~2.500 livros
    │
    ▼
Limpeza e tratamento
Pandas
    │
    ▼
Engenharia de métricas
Nota ponderada + faixas analíticas
    │
    ▼
Análise Exploratória
Rankings + distribuições + relações
    │
    ▼
Base preparada para visualização
    │
    ▼
Relatório interativo
Busca + rankings + consulta + links para o Goodreads
```

---

## Tecnologias utilizadas

O projeto foi desenvolvido principalmente com:

* **Python**
* **Requests** — realização das requisições HTTP;
* **Beautiful Soup** — parsing e extração das informações do HTML;
* **Pandas** — organização, limpeza e transformação dos dados;
* **NumPy** — apoio às operações numéricas;
* **Matplotlib** — análise gráfica;
* **Jupyter Notebook** — desenvolvimento e exploração dos dados;
* **HTML/CSS** — construção e apresentação do relatório final.

---

## Estrutura analítica

O fluxo desenvolvido pode ser resumido em quatro etapas principais:

### 1. Aquisição

Coleta automatizada das páginas da lista *Best Books Ever* e extração das informações de cada livro.

### 2. Preparação

Validação, limpeza, padronização dos tipos das variáveis e criação da base analítica.

### 3. Análise

Construção de métricas, rankings e análises exploratórias sobre avaliação, popularidade, autores, votos e posição na lista.

### 4. Comunicação

Transformação dos resultados em um relatório navegável, permitindo que as análises sejam consultadas de maneira mais direta e intuitiva.

---

## Limitações

Como qualquer projeto baseado em web scraping, alguns pontos devem ser considerados.

A coleta depende da estrutura HTML disponibilizada pelo Goodreads e pode exigir ajustes caso o site seja alterado.

Além disso, os dados representam um **recorte da lista no momento em que a coleta foi realizada**. Rankings, avaliações, quantidade de votos e número de avaliações podem mudar ao longo do tempo.

Também foi considerada a obtenção dos gêneros das obras. Como não foi encontrada uma fonte que permitisse associar essas categorias aos livros de forma suficientemente consistente, essa variável foi retirada da análise final em vez de incorporar classificações potencialmente incorretas.

---

## O que este projeto reúne

Mais do que a extração de uma página web, o projeto integra diferentes etapas comuns a um trabalho de análise de dados:

**aquisição de dados → tratamento → construção de métricas → análise exploratória → visualização → comunicação dos resultados.**

O desenvolvimento partiu de uma atividade da disciplina de **Laboratório em Estatística Aplicada**, mas foi expandido para explorar uma aplicação mais completa do processo analítico, desde a construção da própria fonte de dados até a entrega de uma ferramenta final para consulta dos resultados.

---

## Autora

**Isabella Viana Bambirra**
Graduanda em Estatística — Universidade Federal de São Carlos (UFSCar)

Projeto desenvolvido para fins acadêmicos e de portfólio.

