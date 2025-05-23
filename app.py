from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import pickle
import os

app = Flask(__name__)

# 🔗 Caminhos dos arquivos
MODEL_PATH = os.path.join('modelos', 'modelo_academia.pkl')
DATA_PATH = os.path.join('dados', 'dados_academia.csv')

# 🔥 Banco de Dados SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data/banco.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# 🏗️ Tabela de clientes
class Cliente(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    idade = db.Column(db.Integer)
    tempo = db.Column(db.Integer)
    frequencia = db.Column(db.Integer)
    plano = db.Column(db.Integer)  # 0=Basic, 1=Premium
    previsao = db.Column(db.String(100))

# 🔥 Carregar o modelo e o scaler
try:
    with open(MODEL_PATH, 'rb') as f:
        modelo, scaler = pickle.load(f)
    print("✅ Modelo carregado com sucesso.")
except FileNotFoundError:
    print("❌ ERRO: Arquivo do modelo não encontrado.")
    exit()

# 🏠 Home
@app.route('/')
def home():
    return render_template('home.html')

# ❓ Explicação
@app.route('/explicacao')
def explicacao():
    return render_template('explicacao.html')

# 📊 Gráficos
@app.route('/graficos')
def graficos():
    try:
        dados = pd.read_csv(DATA_PATH)

        risco = dados[dados['Status'] == 'Cancelado'].shape[0]
        seguro = dados[dados['Status'] == 'Ativo'].shape[0]

        dados_vis = dados[['Idade', 'Sexo', 'Tempo_meses', 'Frequencia_semanal', 'Status']].values.tolist()

        return render_template('graficos.html', dados=dados_vis, risco=risco, seguro=seguro)

    except Exception as e:
        return f"Erro ao carregar dados: {e}"

# 🔮 Página de previsão
@app.route('/previsao')
def previsao():
    return render_template('index.html')

# 🔮 API de previsão
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)

        resultado = modelo.predict(features_scaled)[0]
        probabilidade = modelo.predict_proba(features_scaled)[0]

        prob_cancelamento = round(probabilidade[1] * 100, 2)

        # 🔥 Salvar no banco de dados
        novo_cliente = Cliente(
            idade=int(data['features'][0]),
            tempo=int(data['features'][1]),
            frequencia=int(data['features'][2]),
            plano=int(data['features'][3]),
            previsao=f"⚠️ {prob_cancelamento}%" if resultado == 1 else f"✅ Seguro ({100 - prob_cancelamento}%)"
        )
        db.session.add(novo_cliente)
        db.session.commit()

        retorno = {
            'cancelamento_previsto': int(resultado),
            'probabilidade_cancelamento': prob_cancelamento
        }

        return jsonify(retorno)

    except Exception as e:
        return jsonify({'erro': str(e)})

# 🧠 Painel de Controle
@app.route('/painel')
def painel():
    clientes = Cliente.query.all()
    total = len(clientes)
    risco = len([c for c in clientes if c.previsao and '⚠️' in c.previsao])
    seguro = len([c for c in clientes if c.previsao and '✅' in c.previsao])

    return render_template('painel.html', clientes=clientes, total=total, risco=risco, seguro=seguro)

# 🚀 Criar banco na primeira execução
with app.app_context():
    if not os.path.exists('data'):
        os.makedirs('data')
    db.create_all()

# 🚀 Rodar o app
if __name__ == '__main__':
    app.run(debug=True)
