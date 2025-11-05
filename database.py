# database.py
import os
import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import time
import urllib.parse

class DatabaseManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.conn = None
        self.connect()
        if self.conn:
            self.init_tables()
    
    def connect(self):
        """Establecer conexión con PostgreSQL en Railway"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Railway provee DATABASE_URL automáticamente
                database_url = os.environ.get('DATABASE_URL')
                
                if not database_url:
                    self.logger.error("DATABASE_URL no encontrada en variables de entorno")
                    # Intentar construir desde variables individuales
                    database_url = self._build_database_url_from_env()
                
                if not database_url:
                    raise ValueError("No se pudo obtener DATABASE_URL")
                
                # Parsear y asegurar formato correcto
                if database_url.startswith('postgres://'):
                    database_url = database_url.replace('postgres://', 'postgresql://', 1)
                
                self.conn = psycopg2.connect(
                    database_url, 
                    cursor_factory=RealDictCursor,
                    connect_timeout=10
                )
                self.logger.info("✅ Conexión a PostgreSQL establecida")
                return
                
            except Exception as e:
                self.logger.error(f"❌ Error conectando a PostgreSQL (intento {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise
    
    def _build_database_url_from_env(self):
        """Construir DATABASE_URL desde variables individuales"""
        try:
            db_user = os.environ.get('PGUSER')
            db_password = os.environ.get('PGPASSWORD') 
            db_host = os.environ.get('PGHOST', 'postgres.railway.internal')
            db_port = os.environ.get('PGPORT', '5432')
            db_name = os.environ.get('PGDATABASE')
            
            if all([db_user, db_password, db_host, db_name]):
                return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            return None
        except Exception as e:
            self.logger.error(f"Error construyendo DATABASE_URL: {e}")
            return None

    def init_tables(self):
        """Crear tablas si no existen"""
        try:
            with self.conn.cursor() as cur:
                # Tabla para estados actuales
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS estados_tramite (
                        id SERIAL PRIMARY KEY,
                        identificador VARCHAR(255) UNIQUE NOT NULL,
                        ultimo_estado TEXT,
                        timestamp BIGINT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Tabla para historial completo
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS historial_verificaciones (
                        id SERIAL PRIMARY KEY,
                        identificador VARCHAR(255) NOT NULL,
                        fecha_hora VARCHAR(255),
                        estado TEXT,
                        exitoso BOOLEAN,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Índices para mejor performance
                cur.execute("CREATE INDEX IF NOT EXISTS idx_identificador ON estados_tramite(identificador)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_historial_identificador ON historial_verificaciones(identificador)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_historial_fecha ON historial_verificaciones(fecha_hora)")
                
            self.conn.commit()
            self.logger.info("✅ Tablas de PostgreSQL inicializadas")
        except Exception as e:
            self.logger.error(f"❌ Error inicializando tablas: {e}")
            self.conn.rollback()
            raise
    
    def guardar_estado(self, identificador, estado):
        """Guardar o actualizar estado actual"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO estados_tramite (identificador, ultimo_estado, timestamp, updated_at)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (identificador) 
                    DO UPDATE SET 
                        ultimo_estado = EXCLUDED.ultimo_estado,
                        timestamp = EXCLUDED.timestamp,
                        updated_at = CURRENT_TIMESTAMP
                """, (identificador, estado, int(time.time())))
            self.conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error guardando estado en DB: {e}")
            self.conn.rollback()
            return False
    
    def cargar_estado_anterior(self, identificador):
        """Cargar estado anterior desde DB"""
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT ultimo_estado FROM estados_tramite WHERE identificador = %s",
                    (identificador,)
                )
                result = cur.fetchone()
                return result['ultimo_estado'] if result else None
        except Exception as e:
            self.logger.error(f"Error cargando estado desde DB: {e}")
            return None
    
    def cargar_historial(self, identificador, limite=1000):
        """Cargar historial de verificaciones"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    SELECT fecha_hora, estado, exitoso 
                    FROM historial_verificaciones 
                    WHERE identificador = %s 
                    ORDER BY fecha_hora DESC 
                    LIMIT %s
                """, (identificador, limite))
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            self.logger.error(f"Error cargando historial desde DB: {e}")
            return []
    
    def registrar_verificacion(self, identificador, estado, exitoso=True):
        """Registrar nueva verificación en historial"""
        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO historial_verificaciones (identificador, fecha_hora, estado, exitoso)
                    VALUES (%s, %s, %s, %s)
                """, (identificador, time.strftime('%Y-%m-%d %H:%M:%S'), estado, exitoso))
            self.conn.commit()
            return True
        except Exception as e:
            self.logger.error(f"Error registrando verificación en DB: {e}")
            self.conn.rollback()
            return False
    
    def close(self):
        """Cerrar conexión"""
        if self.conn:
            self.conn.close()
            self.logger.info("🔌 Conexión a PostgreSQL cerrada")
