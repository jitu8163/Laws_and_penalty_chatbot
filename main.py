import sys

def run_ingestion():
    from engestion import run_ingestion
    run_ingestion()

def run_api():
    import uvicorn
    uvicorn.run("chat:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py ingest   # Build vector database")
        print("  python main.py api      # Run FastAPI server")
        sys.exit(1)

    mode = sys.argv[1].lower()

    if mode == "ingest":
        run_ingestion()
    elif mode == "api":
        run_api()
    else:
        print("Unknown mode:", mode)
