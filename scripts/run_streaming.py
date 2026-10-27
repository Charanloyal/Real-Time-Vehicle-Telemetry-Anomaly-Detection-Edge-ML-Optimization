"""Run the streaming pipeline (Spark when available, otherwise the local engine)."""

from vehicle_anomaly.streaming import main

if __name__ == "__main__":
    raise SystemExit(main())
