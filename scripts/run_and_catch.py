import traceback
import sys

try:
    import train_llama

    train_llama.main()
except Exception as e:
    with open("crash_report.txt", "w", encoding="utf-8") as f:
        traceback.print_exc(file=f)
    print("CRASHED. Wrote crash_report.txt")
    sys.exit(1)
except KeyboardInterrupt:
    print("Interrupted")
