from pathlib import Path

# --- Rutas ---
ROOT = Path(__file__).resolve().parent.parent
CHESSRED_DIR = ROOT / "data" / "chessred"      # dataset original sin procesar
DATA_DIR = ROOT / "data" / "pieces"            # recortes generados por prepare_data
REAL_TEST_DIR = ROOT / "data" / "real_test"    # fotos de tu montaje (solo test)
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "training" / "runs"          # logs y gráficas (ignorado por git)

# --- Clases ---
CLASSES = ["empty",
           "white_pawn", "white_knight", "white_bishop", "white_rook", "white_queen", "white_king",
           "black_pawn", "black_knight", "black_bishop", "black_rook", "black_queen", "black_king"]

# --- Recorte de casillas (se exporta a pieces.json para la app) ---
WARP_SQUARE_PX = 64                            # tamaño de cada casilla tras la homografía
CROP_HEIGHT_SQUARES = 2                        # el recorte se extiende 1 casilla hacia arriba

# --- Entrada de la red ---
INPUT_W, INPUT_H = 64, 128
MEAN = (0.485, 0.456, 0.406)                   # normalización ImageNet
STD = (0.229, 0.224, 0.225)

# --- Preparación de datos ---
MIN_PERSPECTIVE_RATIO = 0.75                   # 1.0 = cenital; menos = más oblicuo
EMPTY_KEEP_RATIO = 0.3                         # fracción de casillas vacías que se conservan
# Split train/val/test: se usa el oficial de ChessReD (por partida)

# --- Modelo ---
MODEL_NAME = "mobilenet_v3_small"

# --- Entrenamiento ---
BATCH_SIZE = 64
EPOCHS_HEAD = 5                                # fase 1: solo la cabeza
EPOCHS_FINETUNE = 30                           # fase 2: fine-tuning completo
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4
PATIENCE = 5                                   # early stopping
NUM_WORKERS = 2
SEED = 42