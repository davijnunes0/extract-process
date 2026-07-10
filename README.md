# Course Extract

Aplicação em Python para extrair dados estruturados de documentos acadêmicos a partir de imagens, usando modelos de IA multimodal. O projeto foi criado para processar diplomas/certificados e retornar campos como curso, nome do titular, instituição emissora e presença de assinaturas.

## Objetivo

O objetivo do projeto é testar estratégias de extração de informações em documentos acadêmicos digitalizados, comparando respostas de modelos e variantes de prompt. A saída já é preparada para avaliação automática, persistência em MongoDB e experimentos futuros de ensemble.

Campos extraídos atualmente:

- `curso`: nome do curso, programa ou área de formação.
- `nome`: nome completo do titular/diplomado/concluinte.
- `instituicao`: instituição de ensino emissora do documento.
- `assinaturas`: indicação visual da existência de assinatura institucional.

## Observação Sobre Dataset

O dataset usado originalmente neste projeto contém imagens de documentos do Instituto Federal. Por isso, ele não deve ser considerado parte pública ou reutilizável do projeto.

Para executar o projeto em outro ambiente, adicione seu próprio dataset de imagens na pasta configurada em `dataset_path` ou altere essa variável no `main.py`.

Exemplo de estrutura esperada:

```text
dataset-images/
  documento_1.jpg
  documento_2.png
  documento_3.webp
```

Extensões suportadas:

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`
- `.bmp`
- `.gif`

## Como Funciona

O fluxo principal é:

1. Lê as imagens do dataset.
2. Converte cada imagem para Base64 ou Data URL.
3. Envia a imagem para um modelo multimodal via cliente HTTP compatível com chat completions.
4. Aplica um prompt específico para o campo desejado.
5. Tenta transformar a resposta do modelo em JSON estruturado.
6. Normaliza os campos retornados.
7. Opcionalmente compara o resultado com um gabarito (`answer_key.js`).
8. Retorna um objeto pronto para logs, avaliação e persistência em MongoDB.

Formato base de cada resultado:

```python
{
    "document_name": "RT_1.jpg",
    "image_path": "C:/.../dataset-images/RT_1.jpg",
    "model": "gemma4:31b",
    "task": "curso",
    "prompt_variant": "detailed_rules",
    "fields": {
        "curso": "Nome do curso extraído"
    },
    "raw_response": "{...}",
    "error": None
}
```

O projeto ainda mantém chaves de compatibilidade com o formato anterior:

```python
{
    "campos": {...},
    "resposta_bruta": "...",
    "erro": None
}
```

## Estrutura do Projeto

```text
course-extract/
  main.py
  answer_key.js
  pyproject.toml
  README.md
  dataset-images/
  source/
    controller/
      execute_extraction.py
      extract_field.py
      extract_course.py
    services/
      ai_client.py
      image_service.py
      evaluation_service.py
    models/
      mongo_client.py
      people_repository.py
    prompts/
      prompt_loader.py
      curso/
        detailed_rules.txt
        zero_shot.txt
      nome/
        detailed_rules.txt
        zero_shot.txt
      instituicao/
        detailed_rules.txt
        zero_shot.txt
      assinaturas/
        detailed_rules.txt
        zero_shot.txt
    utils/
      console_formatter.py
```

## Principais Módulos

### `main.py`

Arquivo de entrada do projeto. Ele instancia o cliente de IA, define caminhos do dataset e do gabarito, chama os fluxos de extração e imprime os resultados.

Também é o melhor local para orquestrar a persistência no MongoDB, chamando um repositório específico de extrações depois que os resultados forem gerados.

### `source/controller/execute_extraction.py`

Define as funções de extração por campo:

- `execute_course_extraction`
- `execute_name_extraction`
- `execute_signatures_extraction`
- `execute_university_extraction`

Esse arquivo não contém mais prompts hardcoded. Ele carrega os prompts usando `load_prompt()`.

### `source/controller/extract_field.py`

Responsável por executar a extração de fato:

- monta a mensagem para o modelo;
- envia imagem + prompt;
- interpreta JSON retornado;
- trata respostas inválidas;
- normaliza campos;
- retorna o resultado no formato padronizado.

### `source/services/ai_client.py`

Cliente HTTP compatível com o padrão:

```python
client.chat.completions.create(...)
```

Suporta:

- token via variável de ambiente;
- token na URL;
- autenticação com email/senha;
- renovação de token em caso de `401`;
- respostas no formato OpenAI-like e Ollama-like.

### `source/services/image_service.py`

Utilitários para imagens:

- conversão para Base64;
- conversão para Data URL;
- iteração ordenada sobre arquivos de imagem.

### `source/services/evaluation_service.py`

Avalia as extrações usando um gabarito em `answer_key.js`.

Métricas utilizadas:

- similaridade textual;
- distância de Levenshtein;
- acurácia;
- total processado;
- total avaliável;
- total de acertos e erros.

### `source/prompts/`

Armazena os prompts separados por campo e variante.

Variantes disponíveis:

- `detailed_rules`: prompt detalhado com regras explícitas.
- `zero_shot`: prompt curto para testar extração sem exemplos e sem muitas regras.

## Prompts

Os prompts são carregados pelo arquivo:

```text
source/prompts/prompt_loader.py
```

Exemplo:

```python
load_prompt(task="curso", variant="zero_shot")
```

Esse comando carrega:

```text
source/prompts/curso/zero_shot.txt
```

Por padrão, as funções usam:

```python
prompt_variant="detailed_rules"
```

Para testar zero-shot:

```python
execute_course_extraction(
    client=client,
    image_paths=iter_image_paths(dataset_path),
    model="gemma4:31b",
    prompt_variant="zero_shot",
)
```

## Configuração

Crie um arquivo `.env` na raiz do projeto com as variáveis necessárias.

Exemplo:

```env
OPENAI_API_BASE=http://localhost:8081
OPENAI_API_USERNAME=seu-email
OPENAI_API_PASSWORD=sua-senha
OPENAI_CHAT_PATH=/ollama/api/chat
OPENAI_TIMEOUT_SECONDS=1800

MONGO_URI=mongodb://localhost:27017
MONGO_DATABASE=course_extract
```

Variáveis usadas pelo cliente de IA:

- `OPENAI_API_BASE`
- `OPENAI_API_USERNAME`
- `OPENAI_API_PASSWORD`
- `OPENAI_BEARER_TOKEN`
- `OPENAI_CHAT_PATH`
- `OPENAI_TIMEOUT_SECONDS`
- `OPENAI_LOGIN_RETRIES`
- `OPENAI_LOGIN_BACKOFF_SEC`
- `OPENAI_401_RENEW_DELAY_SEC`

Variáveis usadas pelo MongoDB:

- `MONGO_URI`
- `MONGO_DATABASE`

## Instalação

Requisitos:

- Python `>=3.13`
- `uv` ou outro gerenciador de ambiente Python
- acesso a um endpoint de IA multimodal
- MongoDB, caso queira persistir os resultados

Instale as dependências:

```powershell
uv sync
```

Ou, usando `pip`:

```powershell
pip install pymongo python-dotenv requests
```

## Execução

Antes de rodar, ajuste em `main.py`:

```python
dataset_path = r"C:\caminho\para\seu\dataset"
answer_key_path = r"C:\caminho\para\answer_key.js"
```

Depois execute:

```powershell
python main.py
```

Para rodar uma extração específica, chame no `main()` uma das funções:

```python
extract_course()
extract_name()
extract_signature()
extract_institution()
```

## Avaliação

O arquivo `answer_key.js` deve conter o gabarito usado para comparar as extrações.

O avaliador carrega o gabarito, compara os campos extraídos e imprime um resumo com acurácia e similaridade média.

Formato esperado pelo avaliador:

```javascript
const answer = [
  {
    "arquivo": "RT_1.jpg",
    "curso": "Nome esperado do curso",
    "nome": "Nome esperado",
    "instituicao": "Instituição esperada"
  }
]
```

## MongoDB

O projeto já possui um handler de conexão em:

```text
source/models/mongo_client.py
```

Atualmente existe também um `PeopleRepository`, usado para testes de CRUD. Para salvar as extrações, o ideal é criar um repositório específico, por exemplo:

```text
source/models/extraction_repository.py
```

Sugestão de collection:

```text
extractions
```

Documento recomendado para persistência:

```python
{
    "document_name": "RT_1.jpg",
    "image_path": "C:/.../RT_1.jpg",
    "task": "curso",
    "prompt_variant": "zero_shot",
    "model": "gemma4:31b",
    "fields": {
        "curso": "Engenharia de Software"
    },
    "raw_response": "{...}",
    "error": None
}
```

O melhor ponto para inserir no MongoDB é no `main.py`, depois que a extração retorna a lista de resultados.

## Próximos Passos

- Criar `ExtractionRepository` para persistir resultados no MongoDB.
- Salvar `task`, `prompt_variant`, `model`, `fields` e `raw_response` em cada extração.
- Rodar comparações entre `zero_shot` e `detailed_rules`.
- Executar o mesmo documento com múltiplos modelos.
- Implementar ensemble usando as respostas salvas no MongoDB.
- Adicionar testes automatizados para parsing, avaliação e carregamento de prompts.
