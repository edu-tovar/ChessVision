"""ChessVision: prueba de cámara + corrección de distorsión."""
import sys
from pathlib import Path

import cv2

# Para que Python encuentre src/chessvision, se ejecute desde donde se ejecute
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from chessvision.camera import open_camera, Undistorter


def main():
    cap = open_camera()                    # Camo, con índice y resolución de config.py
    undistort = Undistorter.from_file()    # lee data/calibration/calibration.npz

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No llegan frames de la cámara.")
            break

        corrected = undistort(frame)

        cv2.imshow("Original", frame)
        cv2.imshow("Corregida", corrected)
        if cv2.waitKey(1) & 0xFF == ord("q"):   # Q para salir
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()