import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)

#abre o arquivo com o modelo treinado

modelo = pickle.load(open('notebook/modelo_tree.pk1','rb'))

# criando as rotas
# testando se a API está online

@app.route("/")

def index():
  return  render_template('index.html')

# Rota de predição
# ML retorna to tipo de cenário

@app.route('/predict', methods=['POST'])
def predict():
  temperature_in =request. form['temperature_in']
  temperature = request. form['temperature']
  humidity = request. form['humidity']
  gas = request. form['gas']

  dados = {
    'temperature_in' : temperature_in,
    'temperature' : temperature,
    'humidity' : humidity,
    'gas' : gas
  }
  #dados = jsonify(temperature_in,temperature,humidity,gas)
  #dados = request.get_json(force=True)
  #predicao = modelo.predict(np.array([temperature_in,temperature,humidity,gas])
  predicao = modelo.predict(np.array([list(dados.values())]))
  resultado = predicao[0]
  #resposta = {'Cenario': int(resultado)}
  if resultado ==0:
    cenario = 'Normal'
  elif resultado == 1:
    cenario = 'Crítico'
  return render_template('index.html', result = cenario)
  # jsonify(resposta)

  ### subindo server

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 5000))
  app.run(host='0.0.0.0', port=port, debug=True)