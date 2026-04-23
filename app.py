import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ── Configuración de modelos disponibles ─────────────────────────────────
# Cada ejercicio define sus variables, rangos y metadatos.
# Para agregar Glucosa o Energía: colocar el .joblib en /models/ y
# descomentar la entrada correspondiente en este diccionario.

EJERCICIOS = {
    "dolar": {
        "nombre":      "Precio del Dólar",
        "descripcion": "Predice el precio del dólar (COP) a partir de indicadores económicos diarios.",
        "modelo_file": "modelo_dolar.joblib",
        "icono":       "💵",
        "unidad":      "COP",
        "variables": [
            {
                "id":          "Dia",
                "label":       "Día",
                "descripcion": "Número de día del período observado",
                "min":         1,
                "max":         500,
                "step":        1,
                "default":     250,
                "tipo":        "int",
            },
            {
                "id":          "Inflacion",
                "label":       "Inflación diaria",
                "descripcion": "Tasa de inflación diaria (ej: 0.020 = 2%)",
                "min":         0.003,
                "max":         0.040,
                "step":        0.001,
                "default":     0.020,
                "tipo":        "float",
            },
            {
                "id":          "Tasa_interes",
                "label":       "Tasa de interés (%)",
                "descripcion": "Tasa de interés diaria en porcentaje",
                "min":         3.5,
                "max":         6.5,
                "step":        0.1,
                "default":     5.0,
                "tipo":        "float",
            },
        ],
    },

    # ── Ejercicio 2: Glucosa (descomentar cuando el modelo esté listo) ──
    "glucosa": {
    "nombre":      "Nivel de Glucosa",
    "descripcion": "Predice el nivel de glucosa en sangre a partir de variables fisiológicas.",
    "modelo_file": "modelo_glucosa.joblib",
    "icono":       "🩸",
    "unidad":      "mg/dL",
    "variables": [
        {
            "id":          "Edad",
            "label":       "Edad",
            "descripcion": "Edad de la persona en años",
            "min":         18,
            "max":         80,
            "step":        1,
            "default":     40,
            "tipo":        "int",
        },
        {
            "id":          "IMC",
            "label":       "Índice de Masa Corporal (IMC)",
            "descripcion": "Relación entre peso y altura (kg/m²)",
            "min":         13.0,
            "max":         35.0,
            "step":        0.1,
            "default":     24.0,
            "tipo":        "float",
        },
        {
            "id":          "Actividad_Fisica",
            "label":       "Nivel de Actividad Física",
            "descripcion": "Nivel de actividad física diaria (escala de 1 a 10)",
            "min":         1,
            "max":         10,
            "step":        1,
            "default":     5,
            "tipo":        "int",
        },
    ],
},

    # ── Ejercicio 3: Energía (descomentar cuando el modelo esté listo) ──
    "energia": {
    "nombre":      "Consumo de Energía",
    "descripcion": "Predice el consumo eléctrico (kWh) a partir de temperatura ambiente, hora del día y día de la semana.",
    "modelo_file": "modelo_energia.joblib",
    "icono":       "⚡",
    "unidad":      "kWh",
    "variables": [
        {
            "id":          "Temperatura",
            "label":       "Temperatura (°C)",
            "descripcion": "Temperatura ambiente en grados Celsius",
            "min":         5.0,
            "max":         45.0,
            "step":        0.1,
            "default":     25.0,
            "tipo":        "float",
        },
        {
            "id":          "Hora",
            "label":       "Hora del día",
            "descripcion": "Hora en formato 24h (1 = 1:00am, 24 = medianoche)",
            "min":         1,
            "max":         24,
            "step":        1,
            "default":     12,
            "tipo":        "int",
        },
        {
            "id":          "Dia_Semana",
            "label":       "Día de la semana",
            "descripcion": "Día numérico (1 = lunes, 7 = domingo)",
            "min":         1,
            "max":         7,
            "step":        1,
            "default":     4,
            "tipo":        "int",
        },
    ],
},
}

# ── Carga de modelos en memoria al iniciar la app ─────────────────────────
MODELOS_CARGADOS = {}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

for key, config in EJERCICIOS.items():
    ruta = os.path.join(BASE_DIR, "models", config["modelo_file"])
    if os.path.exists(ruta):
        MODELOS_CARGADOS[key] = joblib.load(ruta)
        print(f"  [OK] Modelo '{key}' cargado desde {config['modelo_file']}")
    else:
        print(f"  [--] Modelo '{key}' no encontrado: {ruta}")


# ── Rutas ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Página principal: selector de ejercicio."""
    ejercicios_disponibles = {
        k: v for k, v in EJERCICIOS.items() if k in MODELOS_CARGADOS
    }
    return render_template("index.html", ejercicios=ejercicios_disponibles)


@app.route("/predecir/<ejercicio>", methods=["GET"])
def formulario(ejercicio):
    """Muestra el formulario de predicción para un ejercicio."""
    if ejercicio not in EJERCICIOS:
        return render_template("404.html"), 404
    config = EJERCICIOS[ejercicio]
    disponible = ejercicio in MODELOS_CARGADOS
    return render_template(
        "formulario.html",
        ejercicio_id=ejercicio,
        config=config,
        disponible=disponible,
    )


@app.route("/api/predecir/<ejercicio>", methods=["POST"])
def api_predecir(ejercicio):
    """Endpoint JSON que devuelve la predicción."""
    if ejercicio not in MODELOS_CARGADOS:
        return jsonify({"error": "Modelo no disponible"}), 404

    config    = EJERCICIOS[ejercicio]
    modelo    = MODELOS_CARGADOS[ejercicio]
    variables = config["variables"]

    try:
        valores = []
        for var in variables:
            raw = request.json.get(var["id"])
            if raw is None:
                return jsonify({"error": f"Falta el campo '{var['id']}'"}), 400
            valores.append(float(raw))

        X_input     = np.array(valores).reshape(1, -1)
        prediccion  = modelo.predict(X_input)[0]

        return jsonify({
            "prediccion": round(float(prediccion), 2),
            "unidad":     config["unidad"],
            "ejercicio":  config["nombre"],
            "inputs":     dict(zip([v["id"] for v in variables], valores)),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n  Laboratorio de Minería de Datos — Servidor Flask")
    print("  Modelos disponibles:", list(MODELOS_CARGADOS.keys()))
    print("  Abre http://127.0.0.1:5000\n")
    app.run(debug=True)