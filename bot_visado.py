# bot_visado.py
"""
Bot Visado (CRNN integrado) - Monitor 24/7 con notificaciones y resúmenes en español
- Ejecuta monitoreos periódicos (configurable)
- Envía correo inmediato al detectar cambio de estado
- Envía resumen cada 12 horas (HTML)
- Logs en español
- Usa PostgreSQL (Railway) si está habilitado en config.yaml
- Usa Resend (API) para envío de correos (RESEND_API_KEY en ENV)
"""

import os
import time
import base64
import yaml
import logging
import tempfile
import random
import schedule
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import pickle
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageEnhance

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait, Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException, StaleElementReferenceException

# Intentar importar DatabaseManager si existe
try:
    from database import DatabaseManager
    HAS_DB = True
except Exception:
    HAS_DB = False

# -------------------- Config CRNN (dimensiones esperadas) --------------------
IMG_HEIGHT = 50
IMG_WIDTH = 200

# -------------------- Modelo CRNN (igual que en entrenamiento) --------------------
class CRNN(nn.Module):
    def __init__(self, num_chars, hidden_size=64, rnn_hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.ReLU(), nn.MaxPool2d((1, 2))
        )
        self.feature_size = 256 * 6
        self.fc = nn.Sequential(nn.Linear(self.feature_size, hidden_size), nn.ReLU(), nn.Dropout(0.3))
        self.rnn = nn.LSTM(hidden_size, rnn_hidden_size, num_layers=2, bidirectional=True, batch_first=True, dropout=0.25)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_chars + 1)

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(B, W, C * H)
        x = self.fc(x)
        x, _ = self.rnn(x)
        x = self.classifier(x)
        x = x.permute(1, 0, 2)  # (T, B, C)
        return nn.functional.log_softmax(x, dim=2)

# -------------------- Predictor CRNN robusto (simplificado) --------------------
class CRNNPredictor:
    def __init__(self, model_path, mapping_path, device=None, use_half_on_gpu=True):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.use_half_on_gpu = use_half_on_gpu and self.device.type == "cuda"

        # load mapping
        with open(mapping_path, "rb") as f:
            raw = pickle.load(f)
        # normalize to num_to_char
        if isinstance(raw, dict) and 'num_to_char' in raw:
            num_to_char = raw['num_to_char']
        elif isinstance(raw, dict) and 'char_mapping' in raw and 'num_to_char' in raw['char_mapping']:
            num_to_char = raw['char_mapping']['num_to_char']
        elif isinstance(raw, dict) and 'char_to_num' in raw:
            # invert
            num_to_char = {v: k for k, v in raw['char_to_num'].items()}
        else:
            num_to_char = raw
        self.num_to_char = {int(k): v for k, v in num_to_char.items()}

        max_idx = max(self.num_to_char.keys()) if self.num_to_char else 0
        self.model = CRNN(num_chars=max_idx)
        state = torch.load(model_path, map_location=self.device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            state = state['model_state_dict']
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        if self.use_half_on_gpu:
            self.model.half()
        logging.getLogger("BotVisado").info(f"CRNN cargado en {self.device} (half={self.use_half_on_gpu})")

    def preprocess_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("No se pudo cargar imagen para CRNN")
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype(np.float32) / 255.0
        arr = np.expand_dims(img, axis=0)
        arr = np.expand_dims(arr, axis=0)
        tensor = torch.from_numpy(arr)
        if self.use_half_on_gpu:
            tensor = tensor.half()
        return tensor.to(self.device)

    def ctc_decode(self, log_probs):
        probs = log_probs.exp()
        max_probs, indices = torch.max(probs, dim=2)
        T, B = indices.shape
        indices = indices.cpu().numpy()
        max_probs = max_probs.cpu().numpy()
        decodeds, confs = [], []
        for b in range(B):
            seq = []
            scores = []
            last = -1
            for t in range(T):
                idx = int(indices[t, b])
                if idx != 0 and idx != last:
                    ch = self.num_to_char.get(idx, '')
                    if ch:
                        seq.append(ch)
                        scores.append(float(max_probs[t, b]))
                last = idx
            decodeds.append(''.join(seq))
            confs.append(float(np.mean(scores)) if scores else 0.0)
        return decodeds, confs

    def predict(self, image_path):
        x = self.preprocess_image(image_path)
        with torch.no_grad():
            log_probs = self.model(x)  # (T,B,C)
        decodeds, confs = self.ctc_decode(log_probs)
        return decodeds[0], confs[0]

# -------------------- Bot principal --------------------
class BotVisado:
    DEFAULT_MAX_CONCURRENCY = 4
    DEFAULT_SUMMARY_HOURS = 12

    def __init__(self, config_path="config.yaml"):
        self.config = self._cargar_config(config_path)
        self._setup_logging()
        self._cargar_db()
        self._cargar_crnn()
        self.cuentas = self.config.get('cuentas', [])
        if not self.cuentas:
            self.logger.error("No hay cuentas configuradas en config.yaml")
            raise ValueError("No hay cuentas configuradas")
        self.MAX_CONCURRENCIA = int(self.config.get('max_concurrency', self.DEFAULT_MAX_CONCURRENCY))
        self.executor = ThreadPoolExecutor(max_workers=self.MAX_CONCURRENCIA)
        self.MAX_REINTENTOS = int(self.config.get('max_reintentos', 12))
        # scheduling params
        self.interval_hours = float(self.config.get('monitor_interval_hours', 0.5))  # por defecto 30 minutos
        self.summary_hours = float(self.config.get('summary_hours', self.DEFAULT_SUMMARY_HOURS))
        self.resend_api_key = os.environ.get('RESEND_API_KEY')
        # Estado interno
        self.running = False
        # Para rastrear primeras verificaciones
        self.primeras_verificaciones = self._cargar_primeras_verificaciones()
        self.logger.info(f"Bot inicializado: {len(self.cuentas)} cuentas, concurrencia={self.MAX_CONCURRENCIA}, intervalo={self.interval_hours}h, resumen cada {self.summary_hours}h")

    def _cargar_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Falta {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler("bot_visado.log", encoding="utf-8")]
        )
        self.logger = logging.getLogger("BotVisado")

    def _cargar_db(self):
        self.db = None
        db_conf = self.config.get('postgres', {})
        if db_conf.get('enabled', False) and HAS_DB:
            try:
                # DatabaseManager expects env DATABASE_URL configured by Railway
                self.db = DatabaseManager()
                self.logger.info("Conexión a PostgreSQL (Railway) establecida.")
            except Exception as e:
                self.logger.error(f"No se pudo inicializar la base de datos: {e}")
                self.db = None
        else:
            if not HAS_DB:
                self.logger.warning("DatabaseManager no está disponible; operando sin DB (se usarán archivos locales).")
            else:
                self.logger.info("Postgres deshabilitado por config.")

    def _cargar_crnn(self):
        crnn_cfg = self.config.get('crnn', {})
        model_path = crnn_cfg.get('model_path')
        mapping_path = crnn_cfg.get('mapping_path')
        if model_path and mapping_path and os.path.exists(model_path) and os.path.exists(mapping_path):
            try:
                device = crnn_cfg.get('device', None)
                self.crnn = CRNNPredictor(model_path, mapping_path, device=device)
            except Exception as e:
                self.crnn = None
                self.logger.error(f"No se pudo cargar CRNN: {e}")
        else:
            self.crnn = None
            self.logger.warning("No hay modelo CRNN disponible; se usará Tesseract como fallback.")

    def _cargar_primeras_verificaciones(self):
        """Carga el historial de primeras verificaciones desde archivo"""
        try:
            path = "primeras_verificaciones.json"
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return set(json.load(f))
            return set()
        except Exception as e:
            self.logger.warning(f"Error cargando primeras verificaciones: {e}")
            return set()

    def _guardar_primeras_verificaciones(self):
        """Guarda el historial de primeras verificaciones a archivo"""
        try:
            path = "primeras_verificaciones.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(list(self.primeras_verificaciones), f, ensure_ascii=False)
        except Exception as e:
            self.logger.warning(f"Error guardando primeras verificaciones: {e}")

    # ------------------ Selenium / CAPTCHA ------------------
    def inicializar_selenium(self):
        options = webdriver.ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        # Railway requiere chrome + chromedriver build; asumimos disponible
        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 20)
        return driver, wait

    def capturar_captcha(self, driver, wait, identificador=None):
        try:
            el = wait.until(EC.visibility_of_element_located((By.ID, "imagenCaptcha")))
            script = """
            var img = arguments[0];
            var canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth || img.width;
            canvas.height = img.naturalHeight || img.height;
            var ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
            return canvas.toDataURL('image/png');
            """
            data_url = driver.execute_script(script, el)
            b64 = data_url.split(",", 1)[1]
            data = base64.b64decode(b64)
            tmp = os.path.join(tempfile.gettempdir(), f"captcha_{int(time.time()*1000)}.png")
            with open(tmp, "wb") as f:
                f.write(data)
            self.logger.debug(f"Captcha guardado temporalmente: {tmp}")
            return tmp
        except Exception as e:
            self.logger.error(f"Error capturando CAPTCHA: {e}")
            return None

    def resolver_captcha(self, image_path, identificador=None):
        min_len = int(self.config.get('ocr_min_len', 4))
        conf_threshold = float(self.config.get('crnn', {}).get('conf_threshold', 0.5))
        # Intentar CRNN primero
        if self.crnn:
            try:
                pred, conf = self.crnn.predict(image_path)
                if pred and len(pred) >= min_len and conf >= conf_threshold:
                    self.logger.info(f"CRNN -> '{pred}' (conf={conf:.3f})")
                    return pred, 'crnn', conf
                else:
                    self.logger.info(f"CRNN: pred='{pred}' len={len(pred)} conf={conf:.3f} -> fallback a Tesseract")
            except Exception as e:
                self.logger.error(f"Error CRNN: {e}")

        # Fallback Tesseract
        try:
            import pytesseract
            img = Image.open(image_path)
            img = img.resize((img.width * 3, img.height * 3), Image.LANCZOS)
            img = img.convert("L")
            img = ImageEnhance.Contrast(img).enhance(3.0)
            raw = pytesseract.image_to_string(img, config='--oem 3 --psm 8').strip()
            cleaned = ''.join(ch for ch in raw if ch.isalnum())
            if len(cleaned) >= min_len:
                self.logger.info(f"Tesseract -> '{cleaned}'")
                return cleaned, 'tesseract', 0.0
        except Exception as e:
            self.logger.warning(f"Tesseract falló: {e}")

        return "", 'none', 0.0

    # ------------------ DB helpers (con fallback local) ------------------
    def guardar_estado(self, nombre, identificador, estado):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        if self.db:
            ok = self.db.guardar_estado(identificador, estado)
            self.db.registrar_verificacion(identificador, estado, exitoso=True)
            return ok
        else:
            # fallback local: archivo simple + historial append
            try:
                base = os.path.join("estado_local")
                os.makedirs(base, exist_ok=True)
                with open(os.path.join(base, f"{identificador}.txt"), "w", encoding="utf-8") as f:
                    f.write(estado + "\n" + timestamp)
                # append historial
                with open(os.path.join(base, "historial.log"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"nombre": nombre, "identificador": identificador, "fecha_hora": timestamp, "estado": estado, "exitoso": True}, ensure_ascii=False) + "\n")
                return True
            except Exception as e:
                self.logger.error(f"No se pudo guardar estado local: {e}")
                return False

    def cargar_estado_anterior(self, identificador):
        if self.db:
            try:
                return self.db.cargar_estado_anterior(identificador)
            except Exception as e:
                self.logger.error(f"Error cargando estado anterior desde DB: {e}")
                return None
        else:
            try:
                path = os.path.join("estado_local", f"{identificador}.txt")
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        return f.readline().strip()
                return None
            except Exception as e:
                self.logger.error(f"Error cargando estado local: {e}")
                return None

    def es_primer_monitoreo(self, identificador):
        """Verifica si es la primera vez que se monitorea esta cuenta exitosamente"""
        return identificador not in self.primeras_verificaciones

    def marcar_como_monitoreada(self, identificador):
        """Marca una cuenta como monitoreada por primera vez"""
        self.primeras_verificaciones.add(identificador)
        self._guardar_primeras_verificaciones()

    # ------------------ Notificaciones (Resend) ------------------
    def enviar_notificacion(self, asunto, cuerpo_html, destinatario=None, es_html=True):
        # destinatario: si None, usar config.notifications.email_destino
        email_dest = destinatario or self.config.get('notificaciones', {}).get('email_destino')
        if not email_dest:
            self.logger.error("No hay email destino configurado; notificación omitida.")
            return False

        if not self.resend_api_key:
            # fallback: log y guardar archivo
            self.logger.info(f"(SIMULADO) Envío de email a {email_dest} - Asunto: {asunto}")
            try:
                with open("ultimo_resumen_simulado.html", "w", encoding="utf-8") as f:
                    f.write(cuerpo_html if es_html else f"<pre>{cuerpo_html}</pre>")
            except:
                pass
            return False

        # Enviar mediante Resend
        try:
            import requests
            headers = {"Authorization": f"Bearer {self.resend_api_key}", "Content-Type": "application/json"}
            payload = {
                "from": self.config.get('notificaciones', {}).get('email_from', "Bot Visado <no-reply@example.com>"),
                "to": email_dest,
                "subject": asunto,
                "html": cuerpo_html if es_html else f"<pre>{cuerpo_html}</pre>"
            }
            resp = requests.post("https://api.resend.com/emails", headers=headers, json=payload, timeout=30)
            if resp.status_code in (200, 202):
                self.logger.info(f"Correo enviado a {email_dest} (asunto: {asunto})")
                return True
            else:
                self.logger.error(f"Error Resend API: {resp.status_code} - {resp.text}")
                return False
        except Exception as e:
            self.logger.error(f"Excepción al enviar correo via Resend: {e}")
            return False

    def enviar_notificacion_primer_monitoreo(self, nombre, identificador, estado):
        """Envía notificación cuando se monitorea una cuenta por primera vez"""
        asunto = f"✅ Monitoreo iniciado: {nombre} ({identificador})"
        cuerpo = f"""
        <h3>¡Monitoreo iniciado exitosamente!</h3>
        <p>Se ha comenzado a monitorear el trámite de visado para:</p>
        <ul>
            <li><strong>Nombre:</strong> {nombre}</li>
            <li><strong>Identificador:</strong> {identificador}</li>
            <li><strong>Estado inicial:</strong> {estado}</li>
        </ul>
        <p>El bot verificará periódicamente el estado y te notificará de cualquier cambio.</p>
        <p><em>Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        """
        return self.enviar_notificacion(asunto, cuerpo)

    # ------------------ Resumen HTML ------------------
    def generar_html_resumen(self, rows_html, periodo_texto, estadisticas):
        css = """
        body { font-family: Arial, sans-serif; background:#0b1220; color:#f0f6ff; padding:18px; }
        .card { background:#071022; border-radius:10px; padding:14px; box-shadow:0 6px 18px rgba(0,0,0,0.6); }
        table { width:100%; border-collapse:collapse; margin-top:10px; }
        th, td { padding:8px; text-align:left; border-bottom:1px solid rgba(255,255,255,0.06); font-size:13px; }
        th { color:#9fb3d6; }
        .ok { color:#a7f3d0; font-weight:600 }
        .err { color:#fecaca; font-weight:600 }
        .stats { background: rgba(255,255,255,0.05); padding: 12px; border-radius: 6px; margin: 12px 0; }
        .stat-item { display: inline-block; margin-right: 20px; }
        .stat-value { font-size: 18px; font-weight: bold; }
        """
        
        stats_html = f"""
        <div class="stats">
            <div class="stat-item">
                <div>Monitoreos exitosos</div>
                <div class="stat-value" style="color:#a7f3d0;">{estadisticas['exitosos']}</div>
            </div>
            <div class="stat-item">
                <div>Errores</div>
                <div class="stat-value" style="color:#fecaca;">{estadisticas['errores']}</div>
            </div>
            <div class="stat-item">
                <div>Total monitoreos</div>
                <div class="stat-value" style="color:#9fb3d6;">{estadisticas['total']}</div>
            </div>
            <div class="stat-item">
                <div>Tasa de éxito</div>
                <div class="stat-value" style="color:#93c5fd;">{estadisticas['tasa_exito']}%</div>
            </div>
        </div>
        """
        
        html = f"""<html><head><meta charset="utf-8"><style>{css}</style></head><body>
        <div class="card">
          <h2>📊 Resumen de monitoreo</h2>
          <div style="color:#9fb3d6; font-size:13px;">{periodo_texto}</div>
          {stats_html}
          <table role="presentation">
            <thead><tr><th>Hora</th><th>Cuenta</th><th>Estado</th><th>Resultado</th></tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
          <div style="margin-top:12px; font-size:12px; color:#93b0d6;">Enviado por Bot Visado • {time.strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div></body></html>"""
        return html

    def enviar_resumen_12h(self):
        # Construir resumen desde DB o logs locales
        try:
            now = datetime.now()
            cutoff = now - timedelta(hours=self.summary_hours)
            rows = []
            estadisticas = {"exitosos": 0, "errores": 0, "total": 0, "tasa_exito": 0}
            
            if self.db:
                # cargar historial desde db para cada cuenta
                for c in self.cuentas:
                    ident = c.get('identificador')
                    nombre = c.get('nombre', 'Sin nombre')
                    hist = self.db.cargar_historial(ident, limite=1000)
                    for e in hist:
                        try:
                            fh = e.get('fecha_hora')
                            dt = datetime.strptime(fh, '%Y-%m-%d %H:%M:%S')
                            if dt >= cutoff:
                                resultado = "<span class='ok'>OK</span>" if e.get('exitoso') else "<span class='err'>ERROR</span>"
                                rows.append(f"<tr><td>{fh}</td><td>{nombre} ({ident})</td><td>{e.get('estado')}</td><td>{resultado}</td></tr>")
                                estadisticas['total'] += 1
                                if e.get('exitoso'):
                                    estadisticas['exitosos'] += 1
                                else:
                                    estadisticas['errores'] += 1
                        except Exception:
                            continue
            else:
                # leer historial local
                hist_path = os.path.join("estado_local", "historial.log")
                if os.path.exists(hist_path):
                    with open(hist_path, "r", encoding="utf-8") as f:
                        for line in reversed(f.readlines()):
                            try:
                                obj = json.loads(line.strip())
                                dt = datetime.strptime(obj['fecha_hora'], '%Y-%m-%d %H:%M:%S')
                                if dt >= cutoff:
                                    nombre = obj.get('nombre', 'Sin nombre')
                                    resultado = "<span class='ok'>OK</span>" if obj.get('exitoso') else "<span class='err'>ERROR</span>"
                                    rows.append(f"<tr><td>{obj['fecha_hora']}</td><td>{nombre} ({obj['identificador']})</td><td>{obj['estado']}</td><td>{resultado}</td></tr>")
                                    estadisticas['total'] += 1
                                    if obj.get('exitoso'):
                                        estadisticas['exitosos'] += 1
                                    else:
                                        estadisticas['errores'] += 1
                            except Exception:
                                continue

            # Calcular tasa de éxito
            if estadisticas['total'] > 0:
                estadisticas['tasa_exito'] = round((estadisticas['exitosos'] / estadisticas['total']) * 100, 1)
            
            periodo_texto = f"Resumen desde {cutoff.strftime('%Y-%m-%d %H:%M:%S')} hasta {now.strftime('%Y-%m-%d %H:%M:%S')}"
            html = self.generar_html_resumen(
                ''.join(rows) or "<tr><td colspan='4' style='color:#9fb3d6;padding:12px;'>No hubo actividad en el periodo.</td></tr>", 
                periodo_texto, 
                estadisticas
            )
            asunto = f"📊 Resumen de Monitoreo - Últimas {int(self.summary_hours)}h (Éxito: {estadisticas['tasa_exito']}%)"
            self.enviar_notificacion(asunto, html, destinatario=self.config.get('notificaciones', {}).get('email_destino'))
            self.logger.info(f"Resumen enviado. Estadísticas: {estadisticas['exitosos']} exitosos, {estadisticas['errores']} errores, {estadisticas['tasa_exito']}% tasa de éxito")
        except Exception as e:
            self.logger.error(f"Error generando/enviando resumen: {e}")

    # ------------------ Flujo por cuenta (monitoreo) ------------------
    def consultar_estado_para_cuenta(self, driver, wait, nombre, identificador, ano_nacimiento):
        for intento in range(1, self.MAX_REINTENTOS + 1):
            self.logger.info(f"[{nombre} ({identificador})] Intento {intento}/{self.MAX_REINTENTOS}")
            try:
                driver.get("https://sutramiteconsular.maec.es/")
                time.sleep(random.uniform(1.5, 3.0))
                captcha_path = self.capturar_captcha(driver, wait, identificador)
                if not captcha_path:
                    time.sleep(1.0)
                    continue

                pred, src, conf = self.resolver_captcha(captcha_path, identificador)
                try:
                    os.remove(captcha_path)
                except Exception:
                    pass

                if not pred:
                    self.logger.warning(f"[{nombre} ({identificador})] No se obtuvo predicción válida; reintentando.")
                    time.sleep(1.5)
                    continue

                # Interactuar formulario
                try:
                    tipo_el = wait.until(EC.element_to_be_clickable((By.ID, "infServicio")))
                    Select(tipo_el).select_by_value("VISADO")
                    id_input = wait.until(EC.presence_of_element_located((By.ID, "txIdentificador")))
                    ano_input = wait.until(EC.presence_of_element_located((By.ID, "txtFechaNacimiento")))
                    captcha_input = wait.until(EC.presence_of_element_located((By.ID, "imgcaptcha")))
                    submit_button = wait.until(EC.element_to_be_clickable((By.ID, "imgVerSuTramite")))
                    id_input.clear(); id_input.send_keys(identificador)
                    ano_input.clear(); ano_input.send_keys(ano_nacimiento)
                    captcha_input.clear(); captcha_input.send_keys(pred)
                    time.sleep(random.uniform(0.4, 1.2))
                    try:
                        submit_button.click()
                    except Exception:
                        driver.execute_script("arguments[0].click();", submit_button)
                except Exception as e:
                    self.logger.error(f"[{nombre} ({identificador})] Error al interactuar con el formulario: {e}")
                    time.sleep(1.0)
                    continue

                # Extraer estado
                try:
                    wait.until(EC.presence_of_element_located((By.ID, "CajaGenerica")))
                    titulo = driver.find_element(By.ID, "ContentPlaceHolderConsulta_TituloEstado").text.strip()
                    desc = driver.find_element(By.ID, "ContentPlaceHolderConsulta_DescEstado").text.strip()
                    estado = f"{titulo} - {desc}"
                    self.logger.info(f"[{nombre} ({identificador})] Estado extraído: {estado}")
                    return estado
                except Exception:
                    # comprobar mensaje de captcha rechazado
                    try:
                        err_el = driver.find_element(By.ID, "CompararCaptcha")
                        if err_el and "no concuerdan con la imagen" in err_el.text.lower():
                            self.logger.warning(f"[{nombre} ({identificador})] El servidor indica que el CAPTCHA no coincide.")
                            # registrar intento fallido
                            if self.db:
                                self.db.registrar_verificacion(identificador, "CAPTCHA_INCORRECTO", False)
                            else:
                                with open(os.path.join("estado_local", "historial.log"), "a", encoding="utf-8") as f:
                                    f.write(json.dumps({"nombre": nombre, "identificador": identificador, "fecha_hora": time.strftime('%Y-%m-%d %H:%M:%S'), "estado": "CAPTCHA_INCORRECTO", "exitoso": False}, ensure_ascii=False) + "\n")
                            time.sleep(1.5)
                            continue
                    except Exception:
                        pass
                    self.logger.warning(f"[{nombre} ({identificador})] No se pudo extraer estado; reintentando.")
                    time.sleep(1.5)
                    continue

            except WebDriverException as e:
                self.logger.critical(f"[{nombre} ({identificador})] Error crítico WebDriver: {e}")
                return None
            except Exception as e:
                self.logger.error(f"[{nombre} ({identificador})] Error inesperado: {e}")
                time.sleep(1.5)
                continue

        self.logger.error(f"[{nombre} ({identificador})] Agotados intentos ({self.MAX_REINTENTOS}) sin éxito")
        return None

    # Worker por cuenta (usa executor)
    def worker_cuenta(self, cuenta):
        nombre = cuenta.get('nombre', 'Sin nombre')
        identificador = cuenta.get('identificador')
        ano_nacimiento = cuenta.get('año_nacimiento') or cuenta.get('ano_nacimiento') or ""
        driver = None
        try:
            driver, wait = self.inicializar_selenium()
            estado_actual = self.consultar_estado_para_cuenta(driver, wait, nombre, identificador, ano_nacimiento)
            if estado_actual is None:
                self.logger.warning(f"[{nombre} ({identificador})] No se obtuvo estado en esta ejecución.")
                return
                
            estado_anterior = self.cargar_estado_anterior(identificador)
            
            # Verificar si es primera vez que se monitorea exitosamente
            if self.es_primer_monitoreo(identificador):
                self.enviar_notificacion_primer_monitoreo(nombre, identificador, estado_actual)
                self.marcar_como_monitoreada(identificador)
                self.logger.info(f"[{nombre} ({identificador})] Primera verificación exitosa - notificación enviada")
            
            # registrar en historial y guardar nuevo estado
            if estado_anterior is None or estado_anterior != estado_actual:
                # cambio detectado -> notificar inmediato
                guardado = self.guardar_estado(nombre, identificador, estado_actual)
                # registrar verificación en DB/historial ya hecho en guardar_estado
                asunto = f"🚨 Cambio de estado detectado: {nombre} ({identificador})"
                cuerpo = f"Se detectó un cambio en el trámite para {nombre} ({identificador}).\n\nEstado anterior: {estado_anterior}\nEstado actual: {estado_actual}\n\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                self.enviar_notificacion(asunto, f"<pre>{cuerpo}</pre>", destinatario=cuenta.get('email_notif') or None, es_html=True)
                self.logger.info(f"[{nombre} ({identificador})] Cambio detectado y notificado.")
            else:
                # si no hubo cambio igual registramos verificación (si DB lo requiere)
                if self.db:
                    self.db.registrar_verificacion(identificador, estado_actual, True)
                else:
                    with open(os.path.join("estado_local", "historial.log"), "a", encoding="utf-8") as f:
                        f.write(json.dumps({"nombre": nombre, "identificador": identificador, "fecha_hora": time.strftime('%Y-%m-%d %H:%M:%S'), "estado": estado_actual, "exitoso": True}, ensure_ascii=False) + "\n")
                self.logger.info(f"[{nombre} ({identificador})] Sin cambios (estado: {estado_actual})")
        except Exception as e:
            self.logger.error(f"[{nombre} ({identificador})] Error en worker: {e}")
        finally:
            try:
                if driver:
                    driver.quit()
            except Exception:
                pass

    # ------------------ Monitoreo programado ------------------
    def ejecutar_monitoreo(self):
        self.logger.info("Iniciando ciclo de monitoreo para todas las cuentas...")
        try:
            list(self.executor.map(self.worker_cuenta, self.cuentas))
            self.logger.info("Ciclo de monitoreo finalizado.")
        except Exception as e:
            self.logger.error(f"Error en ejecución de monitoreo: {e}")

    def iniciar(self):
        # Programar tareas
        self.logger.info("Iniciando scheduler (bot 24/7).")
        schedule.clear()
        schedule.every(self.interval_hours).hours.do(self.ejecutar_monitoreo)
        schedule.every(self.summary_hours).hours.do(self.enviar_resumen_12h)

        # Ejecutar una vez al inicio
        self.ejecutar_monitoreo()
        self.enviar_resumen_12h()

        self.running = True
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(10)  # ciclo de espera (10s)
        except KeyboardInterrupt:
            self.logger.info("Interrupción por teclado; cerrando bot...")
        except Exception as e:
            self.logger.error(f"Error en bucle principal: {e}")
        finally:
            self.logger.info("Apagando executor...")
            try:
                self.executor.shutdown(wait=True)
            except:
                pass
            self.logger.info("Bot detenido.")

# ------------------ Ejecución principal ------------------
if __name__ == "__main__":
    bot = BotVisado("config.yaml")
    bot.iniciar()