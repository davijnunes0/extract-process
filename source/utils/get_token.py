import requests

r = requests.get(
    "https://suap.ifg.edu.br/api/autenticacao/token/",
    auth=("20241060140312", "LuizaAmo01$"),
)
print(r.text)
print(r.status_code)
