"""Start the FastAPI service on 0.0.0.0:$PORT (default 8000)."""

from vehicle_anomaly.api import main

if __name__ == "__main__":
    main()
