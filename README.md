# TechVet — Detecção de Neutrófilos

Sistema de visão computacional para detecção automática de **neutrófilos** em imagens microscópicas veterinárias, utilizando YOLOv8 e FastAPI com frontend em Streamlit.

---

## Demonstração

| Imagem Original | Resultado |
|---|---|
| ![original](docs/original.jpg) | ![resultado](docs/resultado.jpg) |

---

## Métricas do modelo

| Métrica | Valor |
|---|---|
| mAP@50 | **0.962** |
| mAP@50-95 | 0.687 |
| F1 Score | 0.901 |
| Precisão | 0.84–0.94 |
| Recall | 0.86–0.96 |

Modelo base: YOLOv8n — fine-tuned por 100 épocas com dataset Roboflow [techvet v2](https://universe.roboflow.com/hub-nlv7n/techvet/dataset/2) (CC BY 4.0).

---

## Arquitetura

```
tecvet2/
├── app.py              # Frontend Streamlit
├── main.py             # API REST FastAPI
├── predictor.py        # Lógica de inferência (compartilhada)
├── data.yaml           # Configuração do dataset YOLO
├── requirements.txt
├── tests/
│   ├── conftest.py
│   ├── test_predictor.py
│   └── test_api.py
└── runs/detect/train/weights/
    └── best.pt         # Pesos do modelo treinado
```

---

## Como executar

### Pré-requisitos

```bash
pip install -r requirements.txt
```

### Frontend Streamlit (recomendado)

```bash
streamlit run app.py
```

Acesse `http://localhost:8501`, faça upload de uma imagem e ajuste o limiar de confiança na barra lateral.

### API REST (FastAPI)

```bash
uvicorn main:app --reload
```

**Endpoint:** `POST /predict`  
**Input:** imagem JPEG/PNG via form-data  
**Output:** imagem JPEG com bounding boxes + header `X-Neutrophil-Count`

Exemplo com `curl`:
```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@sua_imagem.jpg" \
  --output resultado.jpg
```

### Testes

```bash
pytest tests/ -v
```

---

## Stack

- **Modelo:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **API:** FastAPI + Uvicorn
- **Frontend:** Streamlit
- **Imagens:** OpenCV + Pillow
- **Deep Learning:** PyTorch 2.7

---

## Dataset

- **Fonte:** [Roboflow Universe — techvet v2](https://universe.roboflow.com/hub-nlv7n/techvet/dataset/2)
- **Licença:** CC BY 4.0
- **Total:** 142 imagens (99 treino / 29 validação / 14 teste)
- **Classes:** Neutrofilo, Neutrofilos

---

## Autor

**Diogo Oliveira** — Cientista de Dados  
[Portfólio](https://www.datascienceportfol.io/diogohenrique1334) · [GitHub](https://github.com/Diogohenrique1334)
